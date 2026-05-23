#!/usr/bin/env python3
"""main.py — VPU probe client: connect to Unix sockets, receive binary frames."""

import socket
import struct
import select
import time
import sys


def recv_frame(sock):
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


def connect_socket(path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    sock.setblocking(False)
    return sock


def main():
    slow_path = sys.argv[1] if len(sys.argv) > 1 else "./vpu_slow"
    fast_path = sys.argv[2] if len(sys.argv) > 2 else "./vpu_fast"

    print(f"Connecting to {slow_path} and {fast_path}...")
    slow = connect_socket(slow_path)
    fast = connect_socket(fast_path)
    print("Connected. Waiting for data...\n")

    slow_count = 0
    slow_bytes = 0
    fast_count = 0
    fast_bytes = 0
    fast_interval = 0.2  # 200ms sampling
    last_fast_drain = time.time()

    while True:
        readable, _, _ = select.select([slow, fast], [], [], 0.5)

        for sock in readable:
            data = recv_frame(sock)
            if data is None:
                continue

            if sock is slow:
                slow_count += 1
                slow_bytes += len(data)
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] slow frame#{slow_count}: {len(data)} bytes "
                      f"({len(data)//4} words) | total: {slow_bytes} bytes")
            elif sock is fast:
                now = time.time()
                if now - last_fast_drain >= fast_interval:
                    fast_count += 1
                    fast_bytes += len(data)
                    last_fast_drain = now
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] fast frame#{fast_count}: {len(data)} bytes "
                          f"({len(data)//4} words) | total: {fast_bytes} bytes")


if __name__ == "__main__":
    main()
