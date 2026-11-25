"""Shared, versioned cache of model-input clips (owner instruction, 25 Nov 2026).

One cache per preprocessing version under $D/v2/clip-cache/<PREP_VERSION>/:
  meta.json                 version + fingerprint of the preprocessing source code
  shards/<n>.tar            USTAR shards of treated clips (fixed 4 s crop + RMS norm)
  manifest/<n>.jsonl        one record per source clip: source, source_id, locator,
                            prep_version, eligible (+reason), sha256, shard, offset, size

Keys are (source, source_id); the cache is split-agnostic. Experiments (V2-L, V2-A, V2-B)
choose their own rows, splits, audits and packs and only READ bytes here (sha256-checked).
A shard and its manifest are published atomically (tmp -> rename, shard checksum verified
after the copy), so a resumed run skips every clip already recorded and redoes at most one
unpublished shard. One writer at a time (writer.lock).
"""

import hashlib
import inspect
import io
import json
import os
import shutil
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PREP_VERSION = "fixed4s-rmsnorm-v1"


def fingerprint():
    from . import audio, en_treatment

    parts = [inspect.getsource(f) for f in (audio.decode, audio.prepare, audio.fixed_crop,
                                            en_treatment.apply, en_treatment._wav)]
    return hashlib.sha256((PREP_VERSION + "\n".join(parts)).encode()).hexdigest()


def treat(raw):
    from .audio import fixed_crop
    from .en_treatment import apply

    crop = fixed_crop(raw)
    return apply(crop, "norm") if crop is not None else None


_ZIPS = {}


def _from_zip(args):
    """Worker: read one ZIP member (one open ZIP per process) and treat it."""
    zpath, member = args
    t0 = time.perf_counter()
    if zpath not in _ZIPS:
        _ZIPS[zpath] = zipfile.ZipFile(zpath)
    raw = _ZIPS[zpath].read(member)
    t1 = time.perf_counter()
    try:
        b = treat(raw)
        reason = None if b is not None else "shorter_than_4s_after_trim"
    except Exception as e:  # unreadable audio is recorded, not silently dropped
        b, reason = None, f"error:{type(e).__name__}"
    return b, reason, (t1 - t0, time.perf_counter() - t1)


def _from_bytes(raw):
    t1 = time.perf_counter()
    try:
        b = treat(raw)
        reason = None if b is not None else "shorter_than_4s_after_trim"
    except Exception as e:
        b, reason = None, f"error:{type(e).__name__}"
    return b, reason, (0.0, time.perf_counter() - t1)


class ClipCache:
    def __init__(self, data, version=PREP_VERSION):
        if version != PREP_VERSION:
            raise ValueError("Unknown preprocessing version " + version)
        self.root = Path(data) / "v2" / "clip-cache" / version
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        (self.root / "manifest").mkdir(exist_ok=True)
        meta = self.root / "meta.json"
        fp = fingerprint()
        if meta.exists():
            m = json.loads(meta.read_text())
            if m["fingerprint"] != fp or m["version"] != version:
                raise ValueError("Preprocessing code changed; use a new PREP_VERSION")
        else:
            meta.write_text(json.dumps({"version": version, "fingerprint": fp,
                                        "treatment": "audio.fixed_crop 4 s centre crop, then "
                                        "en_treatment.apply('norm'): x*0.05/rms, 16-bit WAV"}, indent=2))
        self.version = version
        self._handles = {}
        self.entries = self._load()

    def _load(self):
        out = {}
        for p in sorted((self.root / "manifest").glob("*.jsonl")):
            for line in p.read_text().splitlines():
                e = json.loads(line)
                key = (e["source"], e["source_id"])
                if key in out:
                    raise ValueError(f"Duplicate cache record {key}")
                out[key] = e
        return out

    def __contains__(self, key):
        return key in self.entries

    def get(self, source, source_id):
        """Treated bytes, or None if recorded as ineligible. Checksum-verified."""
        e = self.entries[(source, source_id)]
        if not e["eligible"]:
            return None
        path = self.root / "shards" / e["shard"]
        f = self._handles.get(path) or self._handles.setdefault(path, path.open("rb"))
        f.seek(e["offset"])
        b = f.read(e["size"])
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise ValueError(f"Cache checksum mismatch {source}/{source_id}")
        return b

    def _next_shard(self):
        return f"{len(list((self.root / 'manifest').glob('*.jsonl'))):06d}"

    def _publish(self, records, scratch):
        name = self._next_shard()
        local = Path(scratch) / f"cache-{name}.tar"
        manifest = []
        with tarfile.open(local, "w", format=tarfile.USTAR_FORMAT) as tar:
            for (source, source_id, locator), b, reason in records:
                e = {"source": source, "source_id": source_id, "locator": locator,
                     "prep_version": self.version, "eligible": b is not None, "reason": reason}
                if b is not None:
                    info = tarfile.TarInfo(hashlib.sha256(f"{source}/{source_id}".encode()).hexdigest()[:32])
                    info.size = len(b)
                    e.update(sha256=hashlib.sha256(b).hexdigest(), shard=name + ".tar",
                             offset=tar.offset + 512, size=len(b))
                    tar.addfile(info, io.BytesIO(b))
                manifest.append(e)
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        dest = self.root / "shards" / (name + ".tar")
        tmp = dest.with_suffix(".tar.tmp")
        shutil.copyfile(local, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != digest:
            raise ValueError("Cache shard copy differs")
        os.replace(tmp, dest)
        local.unlink()
        mtmp = self.root / "manifest" / (name + ".jsonl.tmp")
        mtmp.write_text("".join(json.dumps(e) + "\n" for e in manifest))
        os.replace(mtmp, self.root / "manifest" / (name + ".jsonl"))
        for e in manifest:
            self.entries[(e["source"], e["source_id"])] = e
        return name

    def ensure_zip(self, source, zpath, items, scratch, workers, shard_size=1000, log=print):
        """items: [(source_id, member)]. Treat and publish every item not yet cached.
        Returns timing stats for the newly processed clips."""
        todo = [(sid, m) for sid, m in items if (source, sid) not in self.entries]
        stats = {"requested": len(items), "skipped_cached": len(items) - len(todo),
                 "processed": 0, "seconds": 0.0, "read_s": 0.0, "treat_s": 0.0, "publish_s": 0.0}
        t_all = time.time()
        with ProcessPoolExecutor(workers) as ex:
            for i in range(0, len(todo), shard_size):
                chunk = todo[i:i + shard_size]
                res = list(ex.map(_from_zip, [(str(zpath), m) for _, m in chunk], chunksize=4))
                t = time.time()
                self._publish([((source, sid, m), b, r) for (sid, m), (b, r, _) in zip(chunk, res)], scratch)
                stats["publish_s"] += time.time() - t
                stats["read_s"] += sum(x[2][0] for x in res)
                stats["treat_s"] += sum(x[2][1] for x in res)
                stats["processed"] += len(chunk)
                rate = stats["processed"] / (time.time() - t_all)
                log(f"CACHE {stats['processed'] + stats['skipped_cached']}/{len(items)} "
                    f"{rate:.2f} clips/s eta {(len(todo) - stats['processed']) / rate / 60:.1f} min")
        stats["seconds"] = time.time() - t_all
        return stats

    def ensure_stream(self, source, stream, scratch, workers, shard_size=1000, log=print):
        """stream yields (source_id, locator, raw bytes); for sequential archives (ASV tar)."""
        stats = {"processed": 0, "skipped_cached": 0}
        batch = []
        with ProcessPoolExecutor(workers) as ex:
            def flush():
                res = list(ex.map(_from_bytes, [raw for _, _, raw in batch], chunksize=4))
                self._publish([((source, sid, loc), b, r) for (sid, loc, _), (b, r, _) in zip(batch, res)], scratch)
                stats["processed"] += len(batch)
                log(f"CACHE {source} {stats['processed']} new")
                batch.clear()

            for sid, loc, raw in stream:
                if (source, sid) in self.entries:
                    stats["skipped_cached"] += 1
                    continue
                batch.append((sid, loc, raw))
                if len(batch) >= shard_size:
                    flush()
            if batch:
                flush()
        return stats
