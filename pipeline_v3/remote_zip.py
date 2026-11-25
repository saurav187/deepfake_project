"""Read selected members of a (multi-part, zip64) ZIP over HTTP range requests.

parts: ordered list of (url, size) for disks 0..N (e.g. BD.z01 .. BD.z06, BD.zip).
Only the central directory and the wanted members are transferred.
"""

import struct
import time
import urllib.request
import zlib


def _get(url, start, end, tries=5):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise ValueError(f"range not honoured: HTTP {r.status}")
                b = r.read()
            if len(b) != end - start + 1:
                raise ValueError("short range read")
            return b
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    raise err


class RemoteZip:
    def __init__(self, parts):
        self.parts = parts  # [(url, size)]
        url, size = parts[-1]
        tail = _get(url, max(0, size - 65536), size - 1)
        i = tail.rfind(b"PK\x05\x06")
        (_, disk, cd_disk, n_disk, n, cd_size, cd_off, _) = struct.unpack("<4s4H2LH", tail[i:i + 22])
        j = tail.rfind(b"PK\x06\x06")
        if j >= 0 and (cd_off == 0xFFFFFFFF or n == 0xFFFF):
            (_, _, _, _, disk, cd_disk, n_disk, n, cd_size, cd_off) = struct.unpack("<4sQ2H2L4Q", tail[j:j + 56])
        self.n_entries = n
        self.cd_disk, self.cd_off, self.cd_size = cd_disk, cd_off, cd_size

    def read_span(self, disk, offset, length):
        """Bytes [offset, offset+length) starting on `disk`, continuing into later parts."""
        out = []
        while length > 0:
            url, size = self.parts[disk]
            take = min(length, size - offset)
            if take <= 0:
                disk, offset = disk + 1, 0
                continue
            out.append(_get(url, offset, offset + take - 1))
            length -= take
            disk, offset = disk + 1, 0
        return b"".join(out)

    def directory(self, want=None):
        """{name: (disk, local_offset, comp_size, size, method, crc)} for names in `want`
        (a set, a predicate, or None for all)."""
        cd = self.read_span(self.cd_disk, self.cd_off, self.cd_size)
        out, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (_, _, _, _, method, _, _, crc, csize, size, nlen, xlen, clen, disk, _, _, off) = \
                struct.unpack("<4s6H3L5H2L", cd[p:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            if want is None or (want(name) if callable(want) else name in want):
                extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
                csize, size, off, disk = _zip64(extra, csize, size, off, disk)
                out[name] = (disk, off, csize, size, method, crc)
            p += 46 + nlen + xlen + clen
        return out

    def read(self, info):
        disk, off, csize, size, method, crc = info
        head = self.read_span(disk, off, 30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError("bad local header")
        nlen, xlen = struct.unpack("<2H", head[26:30])
        # Local header may sit at the end of a part; compute data position across parts.
        d, o = disk, off + 30 + nlen + xlen
        while o >= self.parts[d][1]:
            o -= self.parts[d][1]
            d += 1
        data = self.read_span(d, o, csize)
        if method == 8:
            data = zlib.decompressobj(-15).decompress(data)
        elif method != 0:
            raise ValueError(f"unsupported compression {method}")
        if len(data) != size or (zlib.crc32(data) & 0xFFFFFFFF) != crc:
            raise ValueError("member CRC/size mismatch")
        return data


def _zip64(extra, csize, size, off, disk):
    q = 0
    while q + 4 <= len(extra):
        hid, hlen = struct.unpack("<2H", extra[q:q + 4])
        if hid == 1:
            body, r = extra[q + 4:q + 4 + hlen], 0
            if size == 0xFFFFFFFF:
                size = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
            if off == 0xFFFFFFFF:
                off = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
            if disk == 0xFFFF:
                disk = struct.unpack("<L", body[r:r + 4])[0]
            break
        q += 4 + hlen
    return csize, size, off, disk
