"""
04_realtime_inference.py
========================
Real-time deepfake call detector.

Reads audio from a microphone (or a WAV/MP3 file for testing),
processes it in overlapping chunks, and prints a live verdict.

Architecture of the live pipeline
----------------------------------

    Microphone / File
          │
          ▼
    RingBuffer  (accumulates 1 second of audio)
          │
          ▼
    log-Mel spectrogram  (128 × 94, same params as training)
          │
          ▼
    CNNLSTMDetector.forward()  →  raw_score, raw_logit
          │
          ▼
    ThresholdStabilizer  →  chunk_decision, confidence
          │
          ▼
    TemporalSmoother (EMA, window=5)  →  smoothed_score
          │
          ▼
    Console / callback  →  "REAL" / "FAKE" + confidence

Usage
-----
    # Test on a WAV file (no microphone needed)
    python 04_realtime_inference.py --checkpoint outputs/deepfake_cnnlstm.pth
                                    --input_file path/to/test.wav

    # Live microphone mode
    python 04_realtime_inference.py --checkpoint outputs/deepfake_cnnlstm.pth
                                    --mode mic

    # Batch evaluate a folder of files
    python 04_realtime_inference.py --checkpoint outputs/deepfake_cnnlstm.pth
                                    --mode folder
                                    --input_dir path/to/audio/folder
                                    --output_csv results.csv

Requirements
------------
    pip install torch librosa soundfile numpy scipy scikit-learn
    pip install sounddevice   # only for microphone mode
"""

import os
import sys
import csv
import time
import queue
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.special import expit as sigmoid_fn

# ── Internal imports ──────────────────────────────────────────────────────────
# We inline the model definition so this script can run standalone
# without importing the training file.


# ─────────────────────────────────────────────────────────────────────────────
#  Model (must match training definition exactly)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) if pool else nn.Identity(),
        )
    def forward(self, x):
        return self.layers(x)


class CNNLSTMDetector(nn.Module):
    def __init__(self, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.block1 = ConvBlock(1,   32, pool=True)
        self.block2 = ConvBlock(32,  64, pool=True)
        self.block3 = ConvBlock(64, 128, pool=True)
        self.ap     = nn.AdaptiveAvgPool2d((8, 8))
        lstm_input_size = 128 * 8
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.ap(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, C * W)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction  (identical to training)
# ─────────────────────────────────────────────────────────────────────────────
SR         = 16_000
N_MELS     = 128
N_FRAMES   = 94
WIN_LENGTH = int(0.025 * SR)    # 400 samples
HOP_LENGTH = int(0.010 * SR)    # 160 samples
CHUNK_SECS = 1.0                # seconds of audio per chunk
CHUNK_SAMPLES = int(CHUNK_SECS * SR)
DB_MIN     = -80.0
DB_MAX     = 0.0


def audio_to_logmel(audio: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D float32 audio array (16kHz) to a
    (1, 1, N_MELS, N_FRAMES) tensor-ready numpy array.
    """
    import librosa

    # Pad or crop to exactly CHUNK_SAMPLES
    if len(audio) < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
    else:
        audio = audio[:CHUNK_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS,
        n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        fmin=0, fmax=8000, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.clip(log_mel, DB_MIN, DB_MAX)

    T = log_mel.shape[1]
    if T < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, N_FRAMES - T)),
                         constant_values=DB_MIN)
    else:
        log_mel = log_mel[:, :N_FRAMES]

    # (1, 1, 128, 94)
    return log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Threshold Stabilizer (inline, no import needed)
# ─────────────────────────────────────────────────────────────────────────────
class _SimpleStabilizer:
    """Lightweight stabilizer using only what's stored in the checkpoint."""

    def __init__(self, youden_thr: float, temperature: float = 1.0):
        self.youden_thr  = youden_thr
        self.temperature = temperature

    def predict(self, raw_score: float, raw_logit: float) -> tuple[int, float]:
        temp_score = float(sigmoid_fn(raw_logit / self.temperature))
        # Average of raw and temperature-scaled score
        confidence  = (raw_score + temp_score) / 2.0
        # Decision: majority of (Youden on raw, Youden on temp, fixed 0.5)
        v1 = int(raw_score  >= self.youden_thr)
        v2 = int(temp_score >= self.youden_thr)
        v3 = int(confidence >= 0.5)
        decision = int((v1 + v2 + v3) >= 2)
        return decision, confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Smoother (inline)
# ─────────────────────────────────────────────────────────────────────────────
class _EMAsmoother:
    def __init__(self, alpha: float = 0.35, threshold: float = 0.5, window: int = 5):
        self.alpha     = alpha
        self.threshold = threshold
        self.window    = window
        self._ema      = 0.5
        self._n        = 0
        self._buf      = []

    def reset(self):
        self._ema = 0.5; self._n = 0; self._buf = []

    def update(self, score: float) -> tuple[int, float]:
        self._n += 1
        self._buf.append(score)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if self._n == 1:
            self._ema = score
        else:
            self._ema = self.alpha * score + (1.0 - self.alpha) * self._ema
        return int(self._ema >= self.threshold), self._ema

    @property
    def is_warmed_up(self): return self._n >= self.window


# ─────────────────────────────────────────────────────────────────────────────
#  Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeCallDetector:
    """
    End-to-end detector: audio array → verdict.

    Parameters
    ----------
    checkpoint_path : str — path to deepfake_cnnlstm.pth
    device          : torch.device
    window          : number of chunks for temporal smoother
    alpha           : EMA decay
    verbose         : print per-chunk results
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device = None,
        window: int = 5,
        alpha:  float = 0.35,
        verbose: bool = True,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device  = device
        self.verbose = verbose

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Build model
        self.model = CNNLSTMDetector().to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        youden_thr  = ckpt.get("youden_threshold", 0.5)
        temperature = ckpt.get("temperature", 1.0)
        best_auc    = ckpt.get("best_auc", 0.0)
        ir          = ckpt.get("instability_ratio", {})

        print(f"  Model loaded  |  Best AUC={best_auc:.4f}  "
              f"Youden-τ={youden_thr:.4f}  T={temperature:.3f}")
        if ir:
            print(f"  Instability ratio: {ir.get('point',0):.2f}×  "
                  f"[{ir.get('ci_lo',0):.2f}×–{ir.get('ci_hi',0):.2f}×]")

        self.stabilizer = _SimpleStabilizer(youden_thr, temperature)
        self.smoother   = _EMAsmoother(alpha=alpha, threshold=youden_thr, window=window)

    def reset(self):
        """Reset state for a new call."""
        self.smoother.reset()

    @torch.no_grad()
    def _infer_chunk(self, audio_chunk: np.ndarray) -> tuple[float, float]:
        """Run model on a single audio chunk. Returns (raw_score, raw_logit)."""
        feat   = audio_to_logmel(audio_chunk)
        tensor = torch.from_numpy(feat).to(self.device)   # (1,1,128,94)
        logit  = self.model(tensor).item()
        score  = float(sigmoid_fn(logit))
        return score, logit

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a single 1-second audio chunk.

        Returns
        -------
        dict with label, decision, confidence, smoothed_score, is_warmed_up
        """
        raw_score, raw_logit = self._infer_chunk(audio_chunk)
        chunk_dec, confidence = self.stabilizer.predict(raw_score, raw_logit)
        smooth_dec, smooth_score = self.smoother.update(confidence)

        result = {
            "label":          "FAKE" if smooth_dec == 1 else "REAL",
            "decision":        smooth_dec,
            "confidence":      confidence,
            "raw_score":       raw_score,
            "smoothed_score":  smooth_score,
            "is_warmed_up":    self.smoother.is_warmed_up,
            "chunk_n":         self.smoother._n,
        }

        if self.verbose:
            wu = "[warmed]" if result["is_warmed_up"] else "[---]"
            bar_len = int(smooth_score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            color = "\033[91m" if smooth_dec == 1 else "\033[92m"
            reset = "\033[0m"
            print(
                f"  Chunk {result['chunk_n']:3d}  "
                f"{color}[{bar}]{reset}  "
                f"raw={raw_score:.3f}  smooth={smooth_score:.3f}  "
                f"{color}{result['label']}{reset}  {wu}"
            )

        return result

    def process_file(self, file_path: str) -> dict:
        """
        Process an audio file in 1-second non-overlapping chunks.
        Returns the final call verdict.
        """
        import librosa

        self.reset()
        print(f"\nProcessing: {file_path}")
        y, _ = librosa.load(file_path, sr=SR, mono=True)
        n_chunks = max(1, len(y) // CHUNK_SAMPLES)

        results = []
        for i in range(n_chunks):
            chunk = y[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
            res   = self.process_chunk(chunk)
            results.append(res)

        # Final call verdict
        # warmed = [r for r in results if r["is_warmed_up"]] or results
        # n_fake    = sum(1 for r in warmed if r["decision"] == 1)
        # fake_ratio = n_fake / len(warmed)
        # verdict    = "FAKE" if fake_ratio >= 0.40 else "REAL"

        # Final call verdict
        warmed = [r for r in results if r["is_warmed_up"]] or results

        n_fake = sum(1 for r in warmed if r["decision"] == 1)
        fake_ratio = n_fake / len(warmed)

        max_score = max(r["raw_score"] for r in warmed)
        mean_score = np.mean([r["smoothed_score"] for r in warmed])

        # improved decision rule
        if (fake_ratio >= 0.15) or (max_score >= 0.90):
            verdict = "FAKE"
        else:
            verdict = "REAL"

        print(f"\n  ── Final verdict: {verdict}  "
              f"(fake chunks: {n_fake}/{len(warmed)} = {fake_ratio:.0%})")
        print(f"     Mean score: {np.mean([r['smoothed_score'] for r in warmed]):.3f}  "
              f"Max raw score: {max(r['raw_score'] for r in warmed):.3f}")
        return {
            "file":        file_path,
            "verdict":     verdict,
            "fake_ratio":  fake_ratio,
            "n_chunks":    len(results),
            "mean_score":  float(np.mean([r["smoothed_score"] for r in warmed])),
            "max_score":   float(np.max( [r["raw_score"]      for r in warmed])),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Microphone mode
# ─────────────────────────────────────────────────────────────────────────────
def run_microphone_mode(detector: DeepfakeCallDetector, duration_secs: int = 30):
    """
    Listen to the microphone for `duration_secs` seconds,
    processing 1-second chunks in real time.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed.  Run: pip install sounddevice")
        sys.exit(1)

    detector.reset()
    print(f"\n{'='*60}")
    print("LIVE MICROPHONE MODE")
    print(f"Recording for {duration_secs} seconds  (Ctrl-C to stop early)")
    print("=" * 60)

    audio_queue: queue.Queue = queue.Queue()
    ring_buffer = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    results = []
    start   = time.time()

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=int(SR * 0.1),   # 100ms blocks
                        callback=callback):
        while time.time() - start < duration_secs:
            try:
                block = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            ring_buffer = np.concatenate([ring_buffer, block])

            # Process once we have a full 1-second chunk
            while len(ring_buffer) >= CHUNK_SAMPLES:
                chunk       = ring_buffer[:CHUNK_SAMPLES]
                ring_buffer = ring_buffer[CHUNK_SAMPLES:]   # advance (non-overlapping)
                res = detector.process_chunk(chunk)
                results.append(res)

    # Final verdict
    warmed     = [r for r in results if r["is_warmed_up"]] or results
    n_fake     = sum(1 for r in warmed if r["decision"] == 1)
    fake_ratio = n_fake / max(len(warmed), 1)
    verdict    = "FAKE" if fake_ratio >= 0.40 else "REAL"

    print(f"\n{'='*60}")
    print(f"CALL VERDICT: {verdict}")
    print(f"  Fake chunks: {n_fake}/{len(warmed)} ({fake_ratio:.0%})")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch folder mode
# ─────────────────────────────────────────────────────────────────────────────
def run_folder_mode(detector: DeepfakeCallDetector, input_dir: str, output_csv: str):
    """Evaluate all audio files in a folder and save results to CSV."""
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = [p for p in Path(input_dir).rglob("*") if p.suffix.lower() in audio_exts]
    files.sort()

    if not files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"\nBatch evaluation: {len(files)} files  →  {output_csv}")
    rows = []
    for f in files:
        result = detector.process_file(str(f))
        rows.append(result)

    with open(output_csv, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_fake = sum(1 for r in rows if r["verdict"] == "FAKE")
    print(f"\nBatch done: {n_fake}/{len(rows)} files flagged as FAKE")
    print(f"Results saved → {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake Call Detector — Real-Time Inference")
    parser.add_argument("--checkpoint",  type=str,
                        default=str(Path(__file__).resolve().parent / "outputs" / "deepfake_cnnlstm.pth"),
                        help="Path to deepfake_cnnlstm.pth")
    parser.add_argument("--mode",        type=str, default="file",
                        choices=["file", "mic", "folder"],
                        help="Inference mode")
    parser.add_argument("--input_file",  type=str, default=None,
                        help="Path to a single audio file (mode=file)")
    parser.add_argument("--input_dir",   type=str, default=None,
                        help="Directory of audio files (mode=folder)")
    parser.add_argument("--output_csv",  type=str, default=str(Path(__file__).resolve().parent / "results.csv"),
                        help="CSV output path (mode=folder)")
    parser.add_argument("--duration",    type=int, default=30,
                        help="Microphone recording duration in seconds (mode=mic)")
    parser.add_argument("--window",      type=int,   default=5,
                        help="Temporal smoother window size (chunks)")
    parser.add_argument("--alpha",       type=float, default=0.35,
                        help="EMA decay factor")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress per-chunk output")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = DeepfakeCallDetector(
        checkpoint_path = args.checkpoint,
        device  = device,
        window  = args.window,
        alpha   = args.alpha,
        verbose = not args.quiet,
    )

    if args.mode == "file":
        if args.input_file is None:
            print("ERROR: --input_file required for mode=file")
            sys.exit(1)
        detector.process_file(args.input_file)

    elif args.mode == "mic":
        run_microphone_mode(detector, duration_secs=args.duration)

    elif args.mode == "folder":
        if args.input_dir is None:
            print("ERROR: --input_dir required for mode=folder")
            sys.exit(1)
        run_folder_mode(detector, args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
