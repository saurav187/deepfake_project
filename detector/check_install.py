"""Check that this computer can run the released detector and the app.

Usage (repo root):  python -m detector.check_install
Checks ffmpeg, the bundle files and their hashes, the built web page, then loads
the model and analyses a generated test tone (the verdict on a tone means nothing;
it only proves the pipeline runs). Prints how long the analysis took.
"""
import hashlib
import shutil
import sys
from time import perf_counter

from detector import paths

OK, BAD = "  ok  ", "  FAIL"


def main():
    failed = False

    def report(good, text):
        nonlocal failed
        failed |= not good
        print(f"{OK if good else BAD} {text}")

    print(f"Python {sys.version.split()[0]}")
    report(sys.version_info >= (3, 12), "Python 3.12 or newer (3.14 was used to build v1)")
    for tool in ("ffmpeg", "ffprobe"):
        report(shutil.which(tool) is not None, f"{tool} on PATH")

    sums = paths.REPO / "BUNDLE_SHA256SUMS"
    if sums.exists():
        bad = []
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            f = paths.REPO / name.strip()
            if not f.exists():
                bad.append(f"missing {name}")
                continue
            with f.open("rb") as fh:
                if hashlib.file_digest(fh, "sha256").hexdigest() != digest:
                    bad.append(f"changed {name}")
        report(not bad, f"bundle files match BUNDLE_SHA256SUMS{': ' + '; '.join(bad[:5]) if bad else ''}")
    else:
        report(False, "BUNDLE_SHA256SUMS not found: extract deepfake_bundle_v1.tar in the repository root")
    report((paths.REPO / "app/frontend/dist/index.html").exists(), "built web page in app/frontend/dist")
    if failed:
        sys.exit("\nFix the FAIL lines above, then run this again.")

    import numpy as np
    import torch

    from detector.inference import Detector

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    t = perf_counter()
    det = Detector()
    print(f"{OK} model loaded on {det.device} in {perf_counter() - t:.0f} s ({det.info['version']})")
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(10 * 16000) / 16000).astype(np.float32)
    t = perf_counter()
    result = det.analyze_wave(tone)
    print(f"{OK} analysed a 10 s test tone in {perf_counter() - t:.1f} s (verdict: {result['verdict']}; meaningless for a tone)")
    print("\nAll checks passed. Start the app as described in docs/SETUP_LAPTOP.md.")


if __name__ == "__main__":
    main()
