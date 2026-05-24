#!/usr/bin/env python3
"""test_pipeline.py — 后端管线连通性诊断，零外部依赖（仅 stdlib unittest）。

用法:
    make -C hw/gpgpu/platform test-backend
    python3 backend/test_pipeline.py          # 仅单元测试（无需服务器）
    python3 backend/test_pipeline.py --full   # 含集成测试（需启动服务器）
"""

import json
import os
import queue
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.parser.parser import parse_frame

# ── binary frame builder ──────────────────────────────────────────────

INST_ADDI = 0x03130000   # level=0, nargs=3, opcode=19
EVT_REG_WRITE = 0x12010000  # level=1, nargs=2, evt=1


def _header(level, nargs, opcode, branch=0):
    return (level << 28) | (nargs << 24) | (opcode << 16) | branch


def _frame_payload(*records):
    """Build raw payload bytes from (header_word, [operands]) tuples."""
    buf = b""
    for hdr, ops in records:
        buf += struct.pack("<I", hdr)
        for v in ops:
            buf += struct.pack("<I", v)
    return buf


def _socket_frame(payload):
    """Wrap payload with 4-byte LE length prefix (socket wire format)."""
    return struct.pack("<I", len(payload)) + payload


# ── unit tests (no server needed) ─────────────────────────────────────

class TestParser(unittest.TestCase):
    """测试 parse_frame 纯函数 — 二进制 → 结构化 record"""

    def test_single_inst(self):
        hdr = _header(level=0, nargs=3, opcode=19)
        payload = _frame_payload((hdr, [5, 0x2A, 100]))
        records = parse_frame(payload)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["level"], 0)
        self.assertEqual(r["nargs"], 3)
        self.assertEqual(r["opcode"], 19)
        self.assertEqual(r["branch"], False)
        self.assertEqual(r["operands"], [5, 0x2A, 100])

    def test_single_event(self):
        hdr = _header(level=1, nargs=2, opcode=1)
        payload = _frame_payload((hdr, [0x1000, 42]))
        records = parse_frame(payload)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["level"], 1)
        self.assertEqual(r["event"], "REG_WRITE")
        self.assertEqual(r["operands"], [0x1000, 42])

    def test_multi_record(self):
        h1 = _header(level=0, nargs=2, opcode=11)    # LW rd, addr
        h2 = _header(level=1, nargs=1, opcode=4)     # DMA_COMPLETE
        payload = _frame_payload(
            (h1, [3, 0xDEAD]),
            (h2, [0]),  # DMA_COMPLETE status=0 (OK)
        )
        records = parse_frame(payload)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["opcode"], 11)
        self.assertNotIn("event", records[0])  # level=0 无 event 字段
        self.assertEqual(records[1]["opcode"], 4)
        self.assertEqual(records[1]["event"], "DMA_COMPLETE")

    def test_empty_payload(self):
        self.assertEqual(parse_frame(b""), [])

    def test_truncated_header(self):
        self.assertEqual(parse_frame(b"\x01\x02"), [])

    def test_branch_flag(self):
        hdr = _header(level=0, nargs=4, opcode=3, branch=1)  # BEQ
        payload = _frame_payload((hdr, [1, 2, 0x10, 0x1000]))
        records = parse_frame(payload)
        self.assertTrue(records[0]["branch"])


# ── integration tests (need server) ────────────────────────────────────

def _sse_connect_and_read(port, inject_fn, timeout=3.0):
    """在后台线程建立 SSE 连接，确认握手完成后注入数据，读取一条 SSE 消息。
    用 threading.Event 确认 SSE 客户端已注册到 _sse_clients 后再注入。
    """
    import http.client
    result = {"data": None, "error": None}
    connected = threading.Event()

    def _client():
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            conn.request("GET", "/events")
            resp = conn.getresponse()
            if resp.status != 200:
                result["error"] = f"HTTP {resp.status}"
                conn.close()
                connected.set()
                return
            # headers 已收到 → SSE handler 已在 client_q.get() 等待，client 已注册
            connected.set()

            buf = b""
            deadline = time.time() + timeout
            while time.time() < deadline:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                if buf.endswith(b"\n\n"):
                    break
            conn.close()
            text = buf.decode()
            if text.startswith("data: "):
                result["data"] = json.loads(text[len("data: "):].rstrip("\n"))
            elif not text:
                result["error"] = "no data received (timeout)"
            else:
                result["error"] = f"bad SSE: {text[:80]}"
        except Exception as e:
            result["error"] = str(e)
            connected.set()

    t = threading.Thread(target=_client, daemon=True)
    t.start()

    # 等 SSE 握手完成（最长 2 秒）
    if not connected.wait(timeout=2.0):
        return None, "SSE handshake timeout"

    if result["error"]:
        t.join(timeout=1.0)
        return None, result["error"]

    # SSE 已连接，注入数据
    inject_fn()

    t.join(timeout=timeout)
    return result["data"], result["error"]


class TestARawBusToBus(unittest.TestCase):
    """测试 raw_bus → parser → bus（命名以 A 开头确保最先运行，避免 consumer 抢数据）"""

    @classmethod
    def setUpClass(cls):
        import backend.orchestrate.main as m
        cls.m = m
        cls._parser_t = threading.Thread(target=m.parser_thread, daemon=True)
        cls._parser_t.start()

    def test_rawbus_to_bus(self):
        """注入 raw_bus → bus 收到带 ring_type/vpu_id/_seq 的 record"""
        self.assertTrue(self._parser_t.is_alive(), "parser_thread 已死")

        hdr = _header(level=0, nargs=3, opcode=19)
        payload = _frame_payload((hdr, [5, 42, 100]))
        self.m.raw_bus.put(("slow", payload))

        try:
            r = self.m.bus.get(timeout=2.0)
        except queue.Empty:
            self.fail("bus 未收到数据 — parser_thread 未工作")
        self.assertEqual(r["ring_type"], "slow")
        self.assertEqual(r["vpu_id"], 0)
        self.assertIn("_seq", r)
        self.assertEqual(r["opcode"], 19)


class TestZBusToSSE(unittest.TestCase):
    """测试 bus → consumer → SSE（命名 Z 开头确保最后运行，复用前序测试的 parser 线程）"""

    @classmethod
    def setUpClass(cls):
        import backend.orchestrate.main as m
        import socketserver
        cls._m = m

        # 用 ThreadingMixIn 支持多个并发 SSE 连接
        class _ThreadingServer(socketserver.ThreadingMixIn, m.HTTPServer):
            daemon_threads = True

        cls._ServerClass = _ThreadingServer
        cls._ServerClass.allow_reuse_address = True

        # 一个 server 实例服务整个 test class
        cls._port = 18080
        cls._httpd = cls._ServerClass(("127.0.0.1", cls._port), m.RequestHandler)
        cls._srv_t = threading.Thread(target=cls._httpd.serve_forever, daemon=True)
        cls._srv_t.start()

        cls._parser_t = threading.Thread(target=m.parser_thread, daemon=True)
        cls._consumer_t = threading.Thread(target=m.consumer, daemon=True)
        cls._parser_t.start()
        cls._consumer_t.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls._httpd.socket.close()

    def test_bus_to_sse(self):
        """SSE 先连 → 注入 bus → SSE 客户端收到正确 JSON"""
        def _inject():
            self._m.bus.put({
                "vpu_id": 0, "ring_type": "slow", "level": 1,
                "nargs": 2, "opcode": 1, "event": "REG_WRITE",
                "operands": [0x2000, 99], "_seq": 1,
            })

        data, err = _sse_connect_and_read(self._port, _inject, timeout=3.0)
        self.assertIsNone(err, f"SSE error: {err}")
        self.assertIsNotNone(data, "no SSE data")
        self.assertEqual(data["ring_type"], "slow")
        self.assertEqual(data["opcode"], 1)
        self.assertEqual(data["event"], "REG_WRITE")
        self.assertEqual(data["operands"], [0x2000, 99])

    def test_full_pipeline(self):
        """SSE 先连 → raw_bus 注入 → SSE 收到 JSON（端到端）"""
        hdr = _header(level=0, nargs=2, opcode=11)  # LW
        payload = _frame_payload((hdr, [3, 0xBEEF]))

        def _inject():
            self._m.raw_bus.put(("fast", payload))

        data, err = _sse_connect_and_read(self._port, _inject, timeout=3.0)
        self.assertIsNone(err, f"SSE error: {err}")
        self.assertIsNotNone(data, "no SSE data")
        self.assertEqual(data["ring_type"], "fast")
        self.assertEqual(data["opcode"], 11)
        self.assertEqual(data["operands"], [3, 0xBEEF])


# ── mock VPU socket (全链路含 probe) ──────────────────────────────────

class TestZMockSocket(unittest.TestCase):
    """Mock AF_UNIX socket → reader_thread → 全链路（命名 Z 开头确保最后运行）"""

    @classmethod
    def setUpClass(cls):
        import backend.orchestrate.main as m
        import socketserver

        cls._m = m
        cls._port = 18081
        cls._tmpdir = tempfile.mkdtemp()
        cls._sock_path = os.path.join(cls._tmpdir, "vpu_slow")

        cls._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cls._server_sock.bind(cls._sock_path)
        cls._server_sock.listen(1)

        class _ThreadingServer(socketserver.ThreadingMixIn, m.HTTPServer):
            daemon_threads = True
        _ThreadingServer.allow_reuse_address = True
        cls._httpd = _ThreadingServer(("127.0.0.1", cls._port), m.RequestHandler)
        threading.Thread(target=cls._httpd.serve_forever, daemon=True).start()

        threading.Thread(target=m.parser_thread, daemon=True).start()
        threading.Thread(target=m.consumer, daemon=True).start()

        cls._reader_t = threading.Thread(
            target=m.reader_thread, args=("slow", cls._sock_path), daemon=True
        )
        cls._reader_t.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls._httpd.socket.close()
        cls._server_sock.close()
        os.unlink(cls._sock_path)
        os.rmdir(cls._tmpdir)

    def test_socket_to_sse(self):
        """Mock socket 发帧 → SSE 收到数据"""
        self._server_sock.settimeout(2.0)
        try:
            conn, _ = self._server_sock.accept()
        except socket.timeout:
            self.fail("reader_thread 未能连接到 mock socket")
        conn.settimeout(2.0)

        hdr = _header(level=0, nargs=3, opcode=19)  # ADDI
        payload = _frame_payload((hdr, [7, 0x10, 0x20]))
        frame = _socket_frame(payload)

        def _inject():
            conn.sendall(frame)
            conn.close()

        data, err = _sse_connect_and_read(self._port, _inject)
        self.assertIsNone(err, f"mock socket SSE error: {err}")
        self.assertIsNotNone(data, "no SSE data from mock socket")
        self.assertEqual(data["ring_type"], "slow")
        self.assertEqual(data["opcode"], 19)


# ── main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="运行含服务器的集成测试")
    args = ap.parse_args()

    if args.full:
        unittest.main(argv=[sys.argv[0]], defaultTest=None)
    else:
        # 默认只跑单元测试（无需服务器）
        suite = unittest.TestLoader().loadTestsFromTestCase(TestParser)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        print(f"\n✅ 单元测试 {'通过' if result.wasSuccessful() else '失败'}")
        if not result.wasSuccessful():
            print("   运行 --full 进行集成测试")
