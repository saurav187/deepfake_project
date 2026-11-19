"""
03_threshold_stabilizer.py
==========================
Reusable module that implements the threshold stabilization and
temporal smoothing layer described in the project abstract.

This is the engineering realisation of the paper's core finding:
    "AUC is stable, but decision thresholds are not."
    → so we build a stabilized inference module that mitigates drift.

Components
----------
1. ThresholdStabilizer
   - Combines Youden, Platt, and Temperature strategies via majority vote
   - Computes instability ratio on a calibration set
   - Exposes a single .predict(scores) API

2. TemporalSmoother
   - Buffers chunk-level scores from a live call
   - Two modes: majority vote / exponential moving average (EMA)
   - Emits a stable call-level decision

3. StabilizedCallDetector
   - Wraps both components for production use
   - Accepts a raw model score per audio chunk
   - Returns (decision, confidence, smoothed_score)

Usage (standalone)
------------------
    from 03_threshold_stabilizer import StabilizedCallDetector, ThresholdStabilizer

    # Load your trained model checkpoint
    ckpt = torch.load("outputs/deepfake_cnnlstm.pth")

    # Build stabilizer from checkpoint
    stabilizer = ThresholdStabilizer.from_checkpoint(ckpt)

    # Build full call detector (5-chunk window, EMA smoothing)
    detector = StabilizedCallDetector(stabilizer, window=5, method="ema")

    # Feed chunk scores one at a time (from real-time pipeline)
    for score in chunk_scores:
        decision, confidence, smoothed = detector.update(score)
        print(f"Decision: {'FAKE' if decision else 'REAL'}  conf={confidence:.3f}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from scipy.special import expit as sigmoid_fn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
#  Threshold strategies (standalone functions)
# ─────────────────────────────────────────────────────────────────────────────

def youden_optimal(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return Youden-optimal threshold from a calibration set."""
    fpr, tpr, thrs = roc_curve(y_true, scores)
    j = tpr - fpr
    return float(thrs[np.argmax(j)])


def eer_optimal(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return EER threshold via linear interpolation."""
    fpr, tpr, thrs = roc_curve(y_true, scores)
    fnr  = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx  = np.argmin(diff)
    if idx > 0 and diff[idx - 1] < diff[idx]:
        idx -= 1
    return float(thrs[idx])


def platt_calibrate(
    logits_cal: np.ndarray, y_cal: np.ndarray
) -> tuple[LogisticRegression, float]:
    """
    Fit Platt scaling on calibration logits.
    Returns (fitted_lr, youden_threshold_on_calibrated_scores).
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logits_cal.reshape(-1, 1), y_cal)
    platt_scores = lr.predict_proba(logits_cal.reshape(-1, 1))[:, 1]
    fpr, tpr, thrs = roc_curve(y_cal, platt_scores)
    j   = tpr - fpr
    thr = float(thrs[np.argmax(j)])
    return lr, thr


def temperature_calibrate(
    logits_cal: np.ndarray, y_cal: np.ndarray,
    grid: np.ndarray = np.arange(0.1, 5.0, 0.05),
) -> tuple[float, float]:
    """
    Grid-search temperature T to minimise Brier score.
    Returns (T_optimal, youden_threshold_on_temperature_scaled_scores).
    """
    best_T, best_bs = 1.0, float("inf")
    for T in grid:
        probs = sigmoid_fn(logits_cal / T)
        bs    = brier_score_loss(y_cal, probs)
        if bs < best_bs:
            best_bs, best_T = bs, T
    temp_scores = sigmoid_fn(logits_cal / best_T)
    fpr, tpr, thrs = roc_curve(y_cal, temp_scores)
    j   = tpr - fpr
    thr = float(thrs[np.argmax(j)])
    return float(best_T), thr


# ─────────────────────────────────────────────────────────────────────────────
#  ThresholdStabilizer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdStabilizer:
    """
    Combines three threshold strategies via majority vote.

    Parameters
    ----------
    youden_thr  : Raw Youden threshold (on sigmoid scores)
    platt_lr    : Fitted LogisticRegression for Platt scaling
    platt_thr   : Youden threshold on Platt-calibrated scores
    temperature : Optimal temperature T
    temp_thr    : Youden threshold on temperature-scaled scores
    fixed_thr   : Fixed fallback threshold (default 0.5)

    Usage
    -----
        decision, confidence = stabilizer.predict_single(raw_score, raw_logit)
    """
    youden_thr  : float
    platt_lr    : Optional[LogisticRegression] = None
    platt_thr   : float = 0.5
    temperature : float = 1.0
    temp_thr    : float = 0.5
    fixed_thr   : float = 0.5

    # ── calibration-set statistics (populated by .fit) ───────────────
    instability_ratio : float = 0.0
    auc_std           : float = 0.0
    thr_std           : float = 0.0
    ece_raw           : float = 0.0
    ece_platt         : float = 0.0

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "ThresholdStabilizer":
        """
        Reconstruct a ThresholdStabilizer from a saved checkpoint dict.
        The checkpoint is produced by 02_train_cnn_lstm.py.
        Platt lr is NOT serialised in the checkpoint — call .fit() if needed.
        """
        return cls(
            youden_thr  = ckpt.get("youden_threshold", 0.5),
            temperature = ckpt.get("temperature", 1.0),
            temp_thr    = 0.5,
            instability_ratio = ckpt.get("instability_ratio", {}).get("point", 0.0),
        )

    @classmethod
    def fit(
        cls,
        logits_cal: np.ndarray,
        scores_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "ThresholdStabilizer":
        """
        Fit all three strategies on a calibration set.

        Parameters
        ----------
        logits_cal  : Raw model logits on calibration set
        scores_cal  : sigmoid(logits_cal)
        y_cal       : True labels (0=real, 1=fake)
        """
        youden_thr  = youden_optimal(y_cal, scores_cal)
        platt_lr, platt_thr   = platt_calibrate(logits_cal, y_cal)
        temperature, temp_thr = temperature_calibrate(logits_cal, y_cal)

        # ECE for diagnostics
        def ece(y, p, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            e = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (p >= lo) & (p < hi)
                if m.sum(): e += m.sum() * abs(y[m].mean() - p[m].mean())
            return float(e / len(y))

        platt_scores = platt_lr.predict_proba(logits_cal.reshape(-1, 1))[:, 1]
        ece_raw   = ece(y_cal, scores_cal)
        ece_platt = ece(y_cal, platt_scores)

        obj = cls(
            youden_thr  = youden_thr,
            platt_lr    = platt_lr,
            platt_thr   = platt_thr,
            temperature = temperature,
            temp_thr    = temp_thr,
            ece_raw     = ece_raw,
            ece_platt   = ece_platt,
        )
        return obj

    # ── Inference ─────────────────────────────────────────────────────
    def predict_single(
        self,
        raw_score: float,
        raw_logit: Optional[float] = None,
    ) -> tuple[int, float]:
        """
        Predict a single chunk score using majority vote across strategies.

        Parameters
        ----------
        raw_score : sigmoid output of the model (0–1)
        raw_logit : pre-sigmoid logit (required for Platt/Temp scaling)

        Returns
        -------
        (decision, confidence)
        decision   : 0 = real, 1 = fake
        confidence : mean calibrated score across active strategies
        """
        votes   = []
        cal_scores = [raw_score]   # always include raw

        # Strategy 1: Youden on raw score
        votes.append(int(raw_score >= self.youden_thr))

        if raw_logit is not None:
            # Strategy 2: Platt scaling
            if self.platt_lr is not None:
                platt_s = float(self.platt_lr.predict_proba(
                    np.array([[raw_logit]])
                )[0, 1])
                cal_scores.append(platt_s)
                votes.append(int(platt_s >= self.platt_thr))

            # Strategy 3: Temperature scaling
            temp_s = float(sigmoid_fn(raw_logit / self.temperature))
            cal_scores.append(temp_s)
            votes.append(int(temp_s >= self.temp_thr))

        # Majority vote
        decision   = int(sum(votes) > len(votes) / 2)
        confidence = float(np.mean(cal_scores))
        return decision, confidence

    def predict_batch(
        self,
        raw_scores: np.ndarray,
        raw_logits: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch version of predict_single."""
        decisions, confidences = [], []
        logits_iter = raw_logits if raw_logits is not None else [None] * len(raw_scores)
        for s, l in zip(raw_scores, logits_iter):
            d, c = self.predict_single(float(s), float(l) if l is not None else None)
            decisions.append(d); confidences.append(c)
        return np.array(decisions), np.array(confidences)

    def summary(self) -> str:
        lines = [
            "ThresholdStabilizer",
            f"  Youden threshold   : {self.youden_thr:.4f}",
            f"  Platt threshold    : {self.platt_thr:.4f}  "
            f"(Platt fitted: {self.platt_lr is not None})",
            f"  Temperature        : {self.temperature:.3f}",
            f"  Temp threshold     : {self.temp_thr:.4f}",
            f"  ECE (raw)          : {self.ece_raw:.4f}",
            f"  ECE (Platt)        : {self.ece_platt:.4f}",
            f"  Instability ratio  : {self.instability_ratio:.2f}×",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  TemporalSmoother
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalSmoother:
    """
    Smooths per-chunk scores from a live call over a sliding window.

    Parameters
    ----------
    window  : Number of past chunks to consider (default 5 = ~5 seconds)
    method  : "majority" or "ema"
    alpha   : EMA decay factor (only used when method="ema")
    threshold : Decision threshold for smoothed score (default 0.5)

    Reasoning
    ---------
    A single chunk can spike due to background noise, codec artefacts, or
    breath sounds. Smoothing over a window prevents one noisy chunk from
    triggering a false alarm. This directly implements the temporal
    consistency requirement described in the project abstract.
    """
    window    : int   = 5
    method    : Literal["majority", "ema"] = "ema"
    alpha     : float = 0.35          # EMA decay — higher = more reactive
    threshold : float = 0.5

    # Internal state — do not set manually
    _score_buffer : list = field(default_factory=list, repr=False)
    _decision_buffer : list = field(default_factory=list, repr=False)
    _ema           : float = field(default=0.5, repr=False)
    _n_chunks      : int   = field(default=0,   repr=False)

    def reset(self):
        """Call at the start of each new call."""
        self._score_buffer    = []
        self._decision_buffer = []
        self._ema             = 0.5
        self._n_chunks        = 0

    def update(
        self,
        raw_score: float,
        chunk_decision: Optional[int] = None,
    ) -> tuple[int, float, dict]:
        """
        Feed a new chunk score and get a smoothed decision.

        Parameters
        ----------
        raw_score       : sigmoid score from the model (0–1)
        chunk_decision  : optional per-chunk binary decision (from stabilizer)

        Returns
        -------
        (smoothed_decision, smoothed_score, diagnostics)
        smoothed_decision : 0 = real, 1 = fake
        smoothed_score    : float in [0, 1]
        diagnostics       : dict with buffer state for logging
        """
        self._n_chunks += 1
        self._score_buffer.append(raw_score)
        if chunk_decision is not None:
            self._decision_buffer.append(chunk_decision)

        # Keep only the last `window` chunks
        if len(self._score_buffer) > self.window:
            self._score_buffer.pop(0)
        if len(self._decision_buffer) > self.window:
            self._decision_buffer.pop(0)

        if self.method == "majority":
            # Majority vote on the decision buffer
            if len(self._decision_buffer) == 0:
                # Fallback to raw score if no decisions yet
                smoothed_score    = raw_score
                smoothed_decision = int(raw_score >= self.threshold)
            else:
                n_fake = sum(self._decision_buffer)
                n_real = len(self._decision_buffer) - n_fake
                smoothed_decision = int(n_fake > n_real)
                smoothed_score    = float(n_fake / len(self._decision_buffer))

        elif self.method == "ema":
            # EMA on the raw score stream
            if self._n_chunks == 1:
                self._ema = raw_score
            else:
                self._ema = self.alpha * raw_score + (1.0 - self.alpha) * self._ema
            smoothed_score    = self._ema
            smoothed_decision = int(self._ema >= self.threshold)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        diagnostics = {
            "n_chunks":         self._n_chunks,
            "buffer_size":      len(self._score_buffer),
            "buffer_mean":      float(np.mean(self._score_buffer)),
            "ema":              self._ema,
            "smoothed_score":   smoothed_score,
            "smoothed_decision": smoothed_decision,
        }
        return smoothed_decision, smoothed_score, diagnostics

    def is_warmed_up(self) -> bool:
        """True once at least `window` chunks have been processed."""
        return self._n_chunks >= self.window


# ─────────────────────────────────────────────────────────────────────────────
#  StabilizedCallDetector  (combines both components)
# ─────────────────────────────────────────────────────────────────────────────

class StabilizedCallDetector:
    """
    Production-level detector that wraps ThresholdStabilizer + TemporalSmoother.

    Typical usage in a real-time pipeline:

        detector = StabilizedCallDetector.from_checkpoint("deepfake_cnnlstm.pth")
        detector.new_call()

        for audio_chunk in call_stream:
            score, logit = model_inference(audio_chunk)   # your model
            result = detector.update(score, logit)
            if result["is_warmed_up"]:
                print(result["label"], result["confidence"])
    """

    def __init__(
        self,
        stabilizer : ThresholdStabilizer,
        window     : int   = 5,
        method     : str   = "ema",
        alpha      : float = 0.35,
    ):
        self.stabilizer = stabilizer
        self.smoother   = TemporalSmoother(
            window=window, method=method, alpha=alpha,
            threshold=stabilizer.youden_thr,   # reuse Youden threshold for EMA
        )
        self.call_history : list[dict] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        window: int = 5,
        method: str = "ema",
        alpha : float = 0.35,
    ) -> "StabilizedCallDetector":
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        stabilizer = ThresholdStabilizer.from_checkpoint(ckpt)
        return cls(stabilizer, window=window, method=method, alpha=alpha)

    def new_call(self):
        """Reset state at the start of a new call."""
        self.smoother.reset()
        self.call_history.clear()

    def update(self, raw_score: float, raw_logit: Optional[float] = None) -> dict:
        """
        Process one audio chunk.

        Parameters
        ----------
        raw_score : float — sigmoid output of CNN+LSTM
        raw_logit : float — pre-sigmoid logit (optional, enables Platt/Temp)

        Returns
        -------
        dict with keys:
            label          : "FAKE" or "REAL"
            decision       : 0 or 1
            confidence     : calibrated probability estimate
            smoothed_score : EMA / majority-smoothed score
            is_warmed_up   : bool — True after `window` chunks
            chunk_n        : int  — chunk index
            diagnostics    : full smoother state
        """
        # Step 1: Per-chunk stabilized decision
        chunk_decision, confidence = self.stabilizer.predict_single(
            raw_score, raw_logit
        )

        # Step 2: Temporal smoothing
        smoothed_decision, smoothed_score, diag = self.smoother.update(
            confidence, chunk_decision
        )

        result = {
            "label":          "FAKE" if smoothed_decision == 1 else "REAL",
            "decision":        smoothed_decision,
            "confidence":      confidence,
            "raw_score":       raw_score,
            "smoothed_score":  smoothed_score,
            "is_warmed_up":    self.smoother.is_warmed_up(),
            "chunk_n":         self.smoother._n_chunks,
            "diagnostics":     diag,
        }
        self.call_history.append(result)
        return result

    def call_verdict(self) -> dict:
        """
        Final verdict for the completed call.

        Strategy: if >= 40% of warmed-up chunks are FAKE, call is FAKE.
        (Conservative threshold — prefers false alarm over missed detection.)
        """
        if not self.call_history:
            return {"label": "UNKNOWN", "n_chunks": 0}

        warmed = [r for r in self.call_history if r["is_warmed_up"]]
        if not warmed:
            warmed = self.call_history   # use all if call was very short

        n_fake  = sum(1 for r in warmed if r["decision"] == 1)
        n_total = len(warmed)
        fake_ratio = n_fake / n_total
        verdict = 1 if fake_ratio >= 0.40 else 0

        return {
            "label":       "FAKE" if verdict == 1 else "REAL",
            "decision":    verdict,
            "fake_ratio":  fake_ratio,
            "n_chunks":    n_total,
            "mean_score":  float(np.mean([r["smoothed_score"] for r in warmed])),
            "max_score":   float(np.max( [r["raw_score"]      for r in warmed])),
        }

    def __repr__(self):
        return (
            f"StabilizedCallDetector(\n"
            f"  {self.stabilizer.summary()}\n"
            f"  TemporalSmoother(window={self.smoother.window}, "
            f"method={self.smoother.method}, alpha={self.smoother.alpha})\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test / demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ThresholdStabilizer + TemporalSmoother — Demo")
    print("=" * 60)

    np.random.seed(42)
    N_CAL = 200

    # Simulate calibration data
    y_cal       = np.random.randint(0, 2, N_CAL)
    logits_cal  = np.random.randn(N_CAL) + (y_cal * 1.5 - 0.75)
    scores_cal  = sigmoid_fn(logits_cal)

    # Fit stabilizer
    stab = ThresholdStabilizer.fit(logits_cal, scores_cal, y_cal)
    print("\n" + stab.summary())

    # Simulate a 10-chunk fake call
    detector = StabilizedCallDetector(stab, window=5, method="ema")
    detector.new_call()

    print("\nSimulated FAKE call (10 chunks, scores biased high):")
    fake_scores  = sigmoid_fn(np.random.randn(10) + 1.5)   # biased toward fake
    fake_logits  = np.log(fake_scores / (1 - fake_scores + 1e-8))
    for i, (s, l) in enumerate(zip(fake_scores, fake_logits)):
        result = detector.update(float(s), float(l))
        print(f"  Chunk {i+1:2d}  raw={s:.3f}  smoothed={result['smoothed_score']:.3f}"
              f"  → {result['label']}"
              f"{'  [warmed up]' if result['is_warmed_up'] else ''}")

    verdict = detector.call_verdict()
    print(f"\nCall verdict: {verdict['label']}  "
          f"(fake_ratio={verdict['fake_ratio']:.2f}  "
          f"mean_score={verdict['mean_score']:.3f})")

    # Simulate a real call
    detector.new_call()
    print("\nSimulated REAL call (10 chunks, scores biased low):")
    real_scores = sigmoid_fn(np.random.randn(10) - 1.5)   # biased toward real
    real_logits = np.log(real_scores / (1 - real_scores + 1e-8))
    for i, (s, l) in enumerate(zip(real_scores, real_logits)):
        result = detector.update(float(s), float(l))
        print(f"  Chunk {i+1:2d}  raw={s:.3f}  smoothed={result['smoothed_score']:.3f}"
              f"  → {result['label']}"
              f"{'  [warmed up]' if result['is_warmed_up'] else ''}")

    verdict = detector.call_verdict()
    print(f"\nCall verdict: {verdict['label']}  "
          f"(fake_ratio={verdict['fake_ratio']:.2f}  "
          f"mean_score={verdict['mean_score']:.3f})")
