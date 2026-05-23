"""probe.py — VPU socket client: connect, recv binary frames, auto-retry."""

import os
import socket
import struct
import time


def recv_frame(sock):
    """Read one length-prefixed binary frame from socket.

    Wire format: [4B LE uint32 frame_len][frame_len bytes raw data]
    Returns raw bytes or None on disconnect/error.
    """
    raw = sock.recv(4)
    if len(raw) < 4:
        return None
    length = struct.unpack("<I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def run_probe(sock_path, on_frame):
    """Connect to a VPU probe socket, call on_frame(raw_bytes) for each frame.

    Blocks forever. Auto-retries on connection loss. Call from a thread.
    """
    while True:
        sock = None
        try:
            while not os.path.exists(sock_path):
                time.sleep(1)

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)

            while True:
                data = recv_frame(sock)
                if data is None:
                    break
                on_frame(data)
        except (FileNotFoundError, ConnectionRefusedError):
            pass
        finally:
            if sock:
                sock.close()
        time.sleep(2)
