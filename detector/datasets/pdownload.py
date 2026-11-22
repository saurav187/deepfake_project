"""Parallel, resumable, checksum-verified download using HTTP range requests.

Usage: python pdownload.py URL FILE --size BYTES --md5 HEX [--connections 8]

The file is split into 64 MB pieces fetched over several connections at once
and written in place. Finished pieces are recorded in FILE.parts, so a restart
resumes. An existing contiguous prefix (e.g. from an interrupted curl run) is
reused. When every piece is present, the MD5 is checked and FILE.ok is created.
"""
import argparse
import hashlib
import os
from pathlib import Path
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

PIECE = 64 * 1024 * 1024
TIMEOUT = 60          # seconds without data before a connection is retried
USER_AGENT = "dataset-fetch/1.0"


def resolve(url):
    """Follow redirects once so every range request goes to the final host."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.geturl()


def fetch_piece(url, fd, start, end, stats):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if response.status != 206:
            raise RuntimeError(f"server ignored the range request (HTTP {response.status})")
        offset = start
        while offset < end:
            block = response.read(min(1 << 20, end - offset))
            if not block:
                raise RuntimeError("connection closed early")
            os.pwrite(fd, block, offset)
            offset += len(block)
            with stats["lock"]:
                stats["bytes"] += len(block)
    return offset - start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("file", type=Path)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--md5", required=True)
    p.add_argument("--connections", type=int, default=8)
    args = p.parse_args()
    ok = Path(f"{args.file}.ok")
    if ok.exists():
        print(f"already verified: {args.file}")
        return
    parts_file = Path(f"{args.file}.parts")
    n_pieces = (args.size + PIECE - 1) // PIECE
    done = set()
    if parts_file.exists():
        done = {int(x) for x in parts_file.read_text().split()}
    elif args.file.exists() and args.file.stat().st_size < args.size:
        # A sequential downloader (curl) leaves a valid contiguous prefix. A file
        # already at full size without a .parts record is a pre-sized sparse file
        # from an interrupted parallel run and proves nothing, so it is refetched.
        done = set(range(args.file.stat().st_size // PIECE))
    # Record the starting state at once, before the file is pre-sized, so an
    # interruption at any point resumes from the right pieces.
    parts_file.write_text(" ".join(map(str, sorted(done))))
    todo = queue.Queue()
    for i in range(n_pieces):
        if i not in done:
            todo.put(i)
    fd = os.open(args.file, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, args.size)
    url = resolve(args.url)
    stats = {"bytes": 0, "lock": threading.Lock(), "failures": 0}
    done_lock = threading.Lock()
    print(f"{args.file.name}: {len(done)}/{n_pieces} pieces present, {todo.qsize()} to fetch "
          f"over {args.connections} connections", flush=True)

    def worker():
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            start, end = i * PIECE, min(args.size, (i + 1) * PIECE)
            for attempt in range(1, 31):
                try:
                    fetch_piece(url, fd, start, end, stats)
                    break
                except (urllib.error.URLError, OSError, RuntimeError, TimeoutError) as exc:
                    with stats["lock"]:
                        stats["failures"] += 1
                    time.sleep(min(60, 2 * attempt))
            else:
                print(f"giving up on piece {i}", flush=True)
                continue
            with done_lock:
                done.add(i)
                parts_file.write_text(" ".join(map(str, sorted(done))))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.connections)]
    for t in threads:
        t.start()
    last, last_t = 0, time.monotonic()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        now = time.monotonic()
        with stats["lock"]:
            got, failures = stats["bytes"], stats["failures"]
        print(f"{time.strftime('%H:%M:%S')} {args.file.name}: {len(done)}/{n_pieces} pieces, "
              f"{(got - last) / (now - last_t) / 1e6:.2f} MB/s, retries {failures}", flush=True)
        last, last_t = got, now
    os.fsync(fd)
    os.close(fd)
    if len(done) != n_pieces:
        sys.exit(f"{args.file.name}: incomplete ({len(done)}/{n_pieces}); re-run to resume")
    print(f"{args.file.name}: verifying MD5", flush=True)
    digest = hashlib.md5()
    with open(args.file, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    if digest.hexdigest() != args.md5:
        # Drop the record so a re-run cannot trust bad pieces.
        parts_file.unlink(missing_ok=True)
        sys.exit(f"{args.file.name}: MD5 FAILED ({digest.hexdigest()} != {args.md5})")
    ok.touch()
    parts_file.unlink(missing_ok=True)
    print(f"{args.file.name}: OK", flush=True)


if __name__ == "__main__":
    main()
