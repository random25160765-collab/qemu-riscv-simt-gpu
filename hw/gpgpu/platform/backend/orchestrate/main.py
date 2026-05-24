#!/usr/bin/env python3
"""main.py — VPU probe web: probe threads → raw_bus → parser → bus → SSE → browser."""

import json
import os
import queue
import signal
import struct
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(BACKEND_DIR)))

from backend.probe.probe import run_probe
from backend.parser.parser import parse_frame

SCRIPT_DIR = BACKEND_DIR
PLATFORM_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FRONTEND_DIR = os.path.join(PLATFORM_DIR, "frontend")

PROBE_DIR = os.environ.get("VPU_PROBE_DIR", os.path.expanduser("~"))

SOCK_CONFIG = [
    (0, "slow", os.path.join(PROBE_DIR, "vpu_slow")),
    (0, "fast", os.path.join(PROBE_DIR, "vpu_fast")),
]

raw_bus = queue.Queue()
bus = queue.Queue()
_STOP = object()
HTTP_PORT = 8080

_counters = {"raw": 0, "parsed": 0, "sent": 0}
_conn_state = {"slow": "waiting", "fast": "waiting"}

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
}

_sse_lock = threading.Lock()
_sse_clients = []


def _sse_add(client_q):
    with _sse_lock:
        _sse_clients.append(client_q)


def _sse_remove(client_q):
    with _sse_lock:
        _sse_clients.remove(client_q)


def _sse_broadcast(msg):
    data = f"data: {msg}\n\n".encode()
    with _sse_lock:
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/debug-inject":
            self._handle_debug_inject()
        else:
            self.send_error(404)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/status":
            self._handle_status()
        elif path == "/events":
            self._handle_sse()
        elif path == "/test" or path == "/test.html":
            self._serve_static("test.html")
        elif path == "/":
            self._serve_static("index.html")
        elif path == "/style.css":
            self._serve_static("style.css")
        elif path.startswith("/js/"):
            self._serve_static(path.lstrip("/"))
        else:
            self.send_error(404)

    def _handle_debug_inject(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        ring_type = body.get("ring_type", "slow")
        level = body.get("level", 0)
        opcode = body.get("opcode", 19)
        operands = body.get("operands", [0, 0, 0])
        branch = body.get("branch", 0)

        hdr = ((level & 0xF) << 28) | ((len(operands) & 0xF) << 24) | ((opcode & 0xFF) << 16) | (1 if branch else 0)
        payload = struct.pack("<I", hdr)
        for v in operands:
            payload += struct.pack("<I", v)
        raw_bus.put((ring_type, payload))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "ring_type": ring_type, "level": level, "opcode": opcode}).encode())

    def _serve_static(self, rel_path):
        filepath = os.path.join(FRONTEND_DIR, rel_path)
        if not os.path.isfile(filepath):
            self.send_error(404)
            return
        ext = os.path.splitext(rel_path)[1]
        content_type = MIME.get(ext, "application/octet-stream")
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        client_q = queue.Queue(maxsize=256)
        _sse_add(client_q)
        try:
            while True:
                data = client_q.get()
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            _sse_remove(client_q)

    def _handle_status(self):
        import os as _os
        status = {
            "sockets": {
                "slow": {"path": SOCK_CONFIG[0][2], "exists": _os.path.exists(SOCK_CONFIG[0][2])},
                "fast": {"path": SOCK_CONFIG[1][2], "exists": _os.path.exists(SOCK_CONFIG[1][2])},
            },
            "connection": dict(_conn_state),
            "counters": dict(_counters),
            "queues": {
                "raw_bus": raw_bus.qsize(),
                "bus": bus.qsize(),
            },
            "sse_clients": len(_sse_clients),
        }
        body = json.dumps(status, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, fmt, *args):
        pass


def reader_thread(ring_type, sock_path):
    """Socket → raw bytes. Fast ring sampled at 200ms."""

    if ring_type == "fast":
        def on_frame(data):
            nonlocal last_sample
            now = time.time()
            if now - last_sample < 0.2:
                return
            last_sample = now
            _counters["raw"] += 1
            _conn_state[ring_type] = "connected"
            raw_bus.put((ring_type, data))
        last_sample = 0.0
    else:
        def on_frame(data):
            _counters["raw"] += 1
            _conn_state[ring_type] = "connected"
            raw_bus.put((ring_type, data))

    run_probe(sock_path, on_frame)


def parser_thread():
    """raw bytes → structured records."""
    seq = 0
    while True:
        ring_type, data = raw_bus.get()
        records = parse_frame(data)
        _counters["parsed"] += 1
        for r in records:
            r["vpu_id"] = 0
            r["ring_type"] = ring_type
            seq += 1
            r["_seq"] = seq
            bus.put(r)


def consumer():
    """Structured records → JSON → SSE broadcast."""
    while True:
        r = bus.get()
        if r is _STOP:
            break
        _counters["sent"] += 1
        msg = json.dumps({
            "vpu_id":    r["vpu_id"],
            "ring_type": r["ring_type"],
            "level":     r["level"],
            "opcode":    r["opcode"],
            "operands":  r["operands"],
            "event":     r.get("event"),
            "branch":    r.get("branch"),
        })
        _sse_broadcast(msg)


def main():
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("0.0.0.0", HTTP_PORT), RequestHandler)
    print(f"HTTP server listening on http://localhost:{HTTP_PORT}")

    shutdown_flag = threading.Event()

    def _handle_signal(signum, frame):
        print(f"\nSignal {signum} received, shutting down...")
        shutdown_flag.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    threads = []

    for _, ring_type, sock_path in SOCK_CONFIG:
        t = threading.Thread(
            target=reader_thread,
            args=(ring_type, sock_path),
            daemon=True,
        )
        t.start()
        threads.append(t)
        print(f"reader [{ring_type}]: {sock_path}")

    t = threading.Thread(target=parser_thread, daemon=True)
    t.start()
    threads.append(t)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    threads.append(t)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("Ready. Waiting for data...\n")

    try:
        while not shutdown_flag.wait(0.5):
            pass
    finally:
        # socket.close() instead of server.shutdown(): shutdown() waits for
        # all handlers to finish, but SSE handlers block forever on queue.get().
        server.socket.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
