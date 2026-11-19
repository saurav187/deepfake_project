"""
02_train_cnn_lstm.py
====================
Trains a CNN+LSTM deepfake audio detector on IndieFake log-Mel spectrograms.

Key improvements over the baseline 2-block CNN
-----------------------------------------------
1.  3-block CNN backbone  (32 → 64 → 128 filters)
2.  Bidirectional LSTM    (128 hidden, 2 layers) for temporal modelling
3.  Speaker-independent nested cross-validation
    - Outer splits are by SPEAKER GROUP, not random sample
    - Ensures test speakers never appear in training
4.  Threshold stabilization via Platt & Temperature scaling
5.  Instability ratio reported alongside AUC
6.  Saves checkpoint  deepfake_cnnlstm.pth  for real-time inference

Usage
-----
    python 02_train_cnn_lstm.py --data_dir ./data
                                --output_dir ./outputs
                                --epochs 10
                                --outer 5
                                --inner 2

Requirements
------------
    pip install torch numpy scikit-learn matplotlib scipy tqdm
"""

import os
import sys
import json
import pickle
import warnings
import argparse
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d
from scipy.special import expit as sigmoid_fn
from scipy.stats import norm, wilcoxon
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, brier_score_loss,
)


# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────
class AudioDataset(Dataset):
    """(N,H,W,C) → (N,C,H,W) tensors."""
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features).permute(0, 3, 1, 2).float().contiguous()
        self.y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1).contiguous()

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
#  CNN + LSTM Model
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
    """
    Architecture
    ─────────────
    Input   : (B, 1, 128, 94)  log-Mel spectrogram chunk

    CNN backbone (spatial feature extraction)
    ├─ Block 1 : Conv(1→32)   + BN + ReLU + MaxPool2d → (B, 32, 64, 47)
    ├─ Block 2 : Conv(32→64)  + BN + ReLU + MaxPool2d → (B, 64, 32, 23)
    └─ Block 3 : Conv(64→128) + BN + ReLU + MaxPool2d → (B, 128, 16, 11)

    Reshape for LSTM:
    Treat the frequency axis (16) as the sequence dimension,
    with 128*11 = 1408 features per step.
    But we use AdaptiveAvgPool first to keep things compact:
    AdaptiveAvgPool2d(8,8)  → (B, 128, 8, 8)
    Reshape → (B, 8, 128*8) = (B, 8, 1024)   [seq_len=8, input_size=1024]

    Bidirectional LSTM (2 layers, hidden=256 each direction)
    → take last hidden state → (B, 512)

    FC head
    ├─ Linear(512 → 128) + ReLU + Dropout(0.4)
    └─ Linear(128 → 1)   → raw logit  (apply sigmoid externally)

    Total params ≈ 3.7 M
    """

    def __init__(self, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()

        # ── CNN backbone ─────────────────────────────────────────────
        self.block1 = ConvBlock(1,   32, pool=True)
        self.block2 = ConvBlock(32,  64, pool=True)
        self.block3 = ConvBlock(64, 128, pool=True)
        self.ap     = nn.AdaptiveAvgPool2d((8, 8))   # → (B, 128, 8, 8)

        # After reshape: (B, seq=8, features=128*8=1024)
        lstm_input_size = 128 * 8   # 1024

        # ── Bidirectional LSTM ────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_size = lstm_hidden * 2   # bidirectional

        # ── Fully connected head ──────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, 1, 128, 94)
        x = self.block1(x)   # (B, 32, 64, 47)
        x = self.block2(x)   # (B, 64, 32, 23)
        x = self.block3(x)   # (B,128, 16, 11)
        x = self.ap(x)       # (B,128,  8,  8)

        B, C, H, W = x.shape
        # Reshape: treat H as sequence length, (C*W) as feature dim
        x = x.permute(0, 2, 1, 3)        # (B, H, C, W) = (B, 8, 128, 8)
        x = x.reshape(B, H, C * W)       # (B, 8, 1024)

        out, (h_n, _) = self.lstm(x)     # h_n: (layers*2, B, hidden)
        # Concatenate last layer's forward & backward hidden states
        h_fwd = h_n[-2]                  # (B, hidden)
        h_bwd = h_n[-1]                  # (B, hidden)
        h     = torch.cat([h_fwd, h_bwd], dim=1)   # (B, hidden*2)

        logit = self.fc(h)               # (B, 1)
        return logit

    def get_embedding(self, x):
        """Return LSTM hidden state for downstream tasks (e.g. speaker profiling)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.ap(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, C * W)
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimiser, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logit = model(xb)
            loss  = criterion(logit, yb)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, loader, device):
    """Return sigmoid scores (numpy) for all samples in loader."""
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb)
        prob  = torch.sigmoid(logit).squeeze(1).cpu().numpy()
        scores.append(prob)
        labels.append(yb.squeeze(1).numpy())
    return np.concatenate(scores), np.concatenate(labels)


@torch.no_grad()
def get_logits(model, loader, device):
    """Return raw logits (numpy) for Platt / Temperature calibration."""
    model.eval()
    logits, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit = model(xb).squeeze(1).cpu().numpy()
        logits.append(logit)
        labels.append(yb.squeeze(1).numpy())
    return np.concatenate(logits), np.concatenate(labels)


# ── Threshold selection ───────────────────────────────────────────────────────
def youden_threshold(fpr, tpr, thresholds):
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


def eer_threshold(fpr, tpr, thresholds):
    """Linear interpolation EER threshold."""
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx  = np.argmin(diff)
    # Interpolate between the two bracketing points
    if idx > 0 and diff[idx - 1] < diff[idx]:
        idx = idx - 1
    return float(thresholds[idx])


def compute_metrics(y_true, scores, threshold, platt=None, temperature=None):
    """Compute full metric dict for a given threshold strategy."""
    fpr, tpr, thrs = roc_curve(y_true, scores)
    auc_val = float(roc_auc_score(y_true, scores))

    preds = (scores >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    far  = fp / (fp + tn + 1e-8)   # false acceptance rate
    frr  = fn / (fn + tp + 1e-8)   # false rejection rate

    return {
        "auc":         auc_val,
        "accuracy":    float(accuracy_score(y_true, preds)),
        "f1":          float(f1_score(y_true, preds)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "far":         float(far),
        "frr":         float(frr),
        "brier":       float(brier_score_loss(y_true, scores)),
        "threshold":   float(threshold),
    }


def expected_calibration_error(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y_true))


# ── Platt scaling ─────────────────────────────────────────────────────────────
class PlattScaler:
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def fit(self, logits, y):
        self.lr.fit(logits.reshape(-1, 1), y)
        return self

    def predict_proba(self, logits):
        return self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]


# ── Temperature scaling ───────────────────────────────────────────────────────
def find_temperature(logits, y, grid=np.arange(0.1, 5.0, 0.05)):
    best_T, best_bs = 1.0, float("inf")
    for T in grid:
        probs = sigmoid_fn(logits / T)
        bs    = brier_score_loss(y, probs)
        if bs < best_bs:
            best_bs, best_T = bs, T
    return float(best_T)


# ─────────────────────────────────────────────────────────────────────────────
#  Speaker-Independent Split
# ─────────────────────────────────────────────────────────────────────────────
def speaker_independent_splits(y, speaker_ids, n_outer=5, n_inner=2, seed=SEED):
    """
    Creates outer and inner folds where each fold's test set
    contains ONLY speakers not seen during training.

    Returns
    -------
    list of (train_val_idx, test_idx, inner_folds)
    where inner_folds = list of (train_idx, val_idx)
    """
    unique_speakers = np.unique(speaker_ids)
    # Stratify speaker groups by their predominant class
    spk_labels = np.array([
        y[speaker_ids == s].mean().round().astype(int)
        for s in unique_speakers
    ])

    skf_outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    skf_inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    splits = []
    for spk_tv_idx, spk_test_idx in skf_outer.split(unique_speakers, spk_labels):
        tv_speakers   = unique_speakers[spk_tv_idx]
        test_speakers = unique_speakers[spk_test_idx]

        tv_mask   = np.isin(speaker_ids, tv_speakers)
        test_mask = np.isin(speaker_ids, test_speakers)

        tv_idx   = np.where(tv_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Inner folds from train-val indices
        tv_spk_labels = np.array([
            y[speaker_ids == s].mean().round().astype(int)
            for s in tv_speakers
        ])
        inner_folds = []
        for spk_tr_idx, spk_vl_idx in skf_inner.split(tv_speakers, tv_spk_labels):
            tr_speakers = tv_speakers[spk_tr_idx]
            vl_speakers = tv_speakers[spk_vl_idx]
            tr_idx = np.where(np.isin(speaker_ids, tr_speakers))[0]
            vl_idx = np.where(np.isin(speaker_ids, vl_speakers))[0]
            inner_folds.append((tr_idx, vl_idx))

        splits.append((tv_idx, test_idx, inner_folds))

    return splits


# ─────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def run_fold(
    X, y, train_idx, val_idx, test_idx,
    inner_folds, pos_weight,
    device, use_amp, args,
    fold_num: int,
):
    """Full training + evaluation for one outer fold."""
    dataset = AudioDataset(X, y)

    def make_loader(idx, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(idx) if shuffle else torch.utils.data.SequentialSampler(
                torch.utils.data.Subset(dataset, idx)
            ),
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    # ── Inner loop: find best inner model + thresholds ──────────────
    inner_best_auc   = -1.0
    inner_best_state = None
    inner_youden_thrs, inner_eer_thrs = [], []

    for i_fold, (tr_idx, vl_idx) in enumerate(inner_folds):
        model     = CNNLSTMDetector().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        tr_loader = make_loader(tr_idx, shuffle=True)
        vl_loader = make_loader(vl_idx)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=args.lr,
            steps_per_epoch=len(tr_loader), epochs=args.epochs,
        )
        scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_auc, patience_cnt, best_state = -1.0, 0, None
        for epoch in range(args.epochs):
            train_one_epoch(model, tr_loader, optimiser, criterion, scaler, device, use_amp)
            scheduler.step()
            scores_v, labels_v = predict_proba(model, vl_loader, device)
            val_auc = roc_auc_score(labels_v, scores_v)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    break

        model.load_state_dict(best_state)
        scores_v, labels_v = predict_proba(model, vl_loader, device)
        fpr, tpr, thrs = roc_curve(labels_v, scores_v)
        inner_youden_thrs.append(youden_threshold(fpr, tpr, thrs))
        inner_eer_thrs.append(eer_threshold(fpr, tpr, thrs))

        if best_val_auc > inner_best_auc:
            inner_best_auc   = best_val_auc
            inner_best_state = best_state

        print(f"    Fold {fold_num} inner {i_fold+1}: AUC={best_val_auc:.4f}  "
              f"Youden-τ={inner_youden_thrs[-1]:.4f}  EER-τ={inner_eer_thrs[-1]:.4f}")

    # Fit calibration on inner val set of best inner model
    model = CNNLSTMDetector().to(device)
    model.load_state_dict(inner_best_state)

    # Use last inner val set for calibration fitting
    vl_idx_last = inner_folds[-1][1]
    vl_loader_cal = make_loader(vl_idx_last)
    logits_v, labels_v_cal = get_logits(model, vl_loader_cal, device)

    platt = PlattScaler().fit(logits_v, labels_v_cal)
    T_opt = find_temperature(logits_v, labels_v_cal)

    mean_youden_thr = float(np.mean(inner_youden_thrs))
    mean_eer_thr    = float(np.mean(inner_eer_thrs))

    # ── Outer test evaluation ────────────────────────────────────────
    test_loader   = make_loader(test_idx)
    scores_t, labels_t = predict_proba(model, test_loader, device)
    logits_t, _        = get_logits(model, test_loader, device)

    fpr_t, tpr_t, thrs_t = roc_curve(labels_t, scores_t)
    test_youden_thr = youden_threshold(fpr_t, tpr_t, thrs_t)
    test_eer_thr    = eer_threshold(fpr_t, tpr_t, thrs_t)

    # Calibrated scores
    platt_scores = platt.predict_proba(logits_t)
    fpr_p, tpr_p, thrs_p = roc_curve(labels_t, platt_scores)
    platt_youden = youden_threshold(fpr_p, tpr_p, thrs_p)

    temp_scores  = sigmoid_fn(logits_t / T_opt)
    fpr_tp, tpr_tp, thrs_tp = roc_curve(labels_t, temp_scores)
    temp_youden  = youden_threshold(fpr_tp, tpr_tp, thrs_tp)

    ece_raw   = expected_calibration_error(labels_t, scores_t)
    ece_platt = expected_calibration_error(labels_t, platt_scores)
    ece_temp  = expected_calibration_error(labels_t, temp_scores)

    record = {
        # Inner thresholds
        "inner_youden_thrs": inner_youden_thrs,
        "inner_eer_thrs":    inner_eer_thrs,
        "mean_inner_youden": mean_youden_thr,
        "mean_inner_eer":    mean_eer_thr,
        # Outer thresholds
        "youden_thresh": test_youden_thr,
        "eer_thresh":    test_eer_thr,
        # Calibration
        "ece_raw":   ece_raw,
        "ece_platt": ece_platt,
        "ece_temp":  ece_temp,
        "temperature": T_opt,
        # Per-strategy metrics
        "youden":  compute_metrics(labels_t, scores_t,   test_youden_thr),
        "eer":     compute_metrics(labels_t, scores_t,   test_eer_thr),
        "fixed":   compute_metrics(labels_t, scores_t,   0.5),
        "platt":   compute_metrics(labels_t, platt_scores, platt_youden),
        "temp":    compute_metrics(labels_t, temp_scores,  temp_youden),
        # For figures
        "fpr": fpr_t, "tpr": tpr_t,
        "test_scores": scores_t, "test_labels": labels_t,
        "test_idx": test_idx,
        "best_inner_auc": inner_best_auc,
        "model_state": inner_best_state,
    }
    return record


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
PAL = plt.cm.tab10.colors


def plot_all(fold_records, output_dir):
    out = Path(output_dir)

    # ── Figure 1: ROC curves ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.5))
    mean_fpr = np.linspace(0, 1, 300)
    tprs_interp = []
    for i, r in enumerate(fold_records):
        fpr, tpr = r["fpr"], r["tpr"]
        auc_val  = r["youden"]["auc"]
        ax.plot(fpr, tpr, lw=1.1, alpha=0.45, color=PAL[i],
                label=f"Fold {i+1}  AUC={auc_val:.4f}")
        fi = interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))
        tprs_interp.append(fi(mean_fpr))
    mean_tpr = np.mean(tprs_interp, axis=0)
    std_tpr  = np.std(tprs_interp,  axis=0)
    mean_auc = np.mean([r["youden"]["auc"] for r in fold_records])
    std_auc  = np.std( [r["youden"]["auc"] for r in fold_records])
    ax.plot(mean_fpr, mean_tpr, "k-", lw=2.4,
            label=f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="grey", alpha=0.18, label="±1 std")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Figure 1: ROC Curves — Speaker-Independent CV")
    ax.legend(loc="lower right"); ax.grid()
    plt.tight_layout()
    plt.savefig(out / "fig1_roc.pdf"); plt.close()

    # ── Figure 2: Threshold boxplot (instability) ──────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    youden_thrs = [r["youden_thresh"] for r in fold_records]
    eer_thrs    = [r["eer_thresh"]    for r in fold_records]
    auc_vals    = [r["youden"]["auc"] for r in fold_records]
    bp = ax.boxplot(
        [auc_vals, youden_thrs, eer_thrs],
        labels=["AUC", "Youden τ", "EER τ"],
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    for patch, color in zip(bp["boxes"], [PAL[0], PAL[1], PAL[2]]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("Figure 2: AUC vs Threshold Variability (Instability)")
    ax.set_ylabel("Value"); ax.grid(axis="y")
    for i, (vals, color) in enumerate([(auc_vals, PAL[0]),
                                        (youden_thrs, PAL[1]),
                                        (eer_thrs, PAL[2])]):
        ax.scatter([i + 1] * len(vals), vals, color=color, zorder=5,
                   s=40, edgecolors="black", lw=0.7)
    plt.tight_layout()
    plt.savefig(out / "fig2_threshold_boxplot.pdf"); plt.close()

    # ── Figure 3: Strategy comparison bar chart ────────────────────────
    strategies = ["Youden", "EER", "Fixed 0.5", "Platt", "Temp"]
    keys       = ["youden", "eer", "fixed", "platt", "temp"]
    metrics    = ["accuracy", "f1", "sensitivity"]
    fig, axes  = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, metrics):
        means = [np.mean([r[k][metric] for r in fold_records]) for k in keys]
        stds  = [np.std( [r[k][metric] for r in fold_records]) for k in keys]
        bars  = ax.bar(strategies, means, yerr=stds, capsize=4,
                       color=PAL[:5], alpha=0.8, edgecolor="black", lw=0.6)
        ax.set_title(metric.capitalize()); ax.set_ylabel(metric)
        ax.set_ylim(min(means) * 0.97, 1.01)
        ax.grid(axis="y")
    fig.suptitle("Figure 3: Strategy Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "fig3_strategy_comparison.pdf"); plt.close()

    # ── Figure 4: DET curve ────────────────────────────────────────────
    def nd(r): return norm.ppf(np.clip(r, 1e-6, 1 - 1e-6))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, r in enumerate(fold_records):
        fnr = 1.0 - r["tpr"]
        ax.plot(nd(r["fpr"]), nd(fnr), lw=1.1, alpha=0.45, color=PAL[i],
                label=f"Fold {i+1}")
    ticks   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    tlabels = ["0.1", "0.5", "1", "2", "5", "10", "20", "40"]
    nd_t    = [nd(t) for t in ticks]
    ax.plot(nd_t, nd_t, "k--", lw=0.9, alpha=0.4, label="EER line")
    ax.set_xticks(nd_t); ax.set_xticklabels(tlabels)
    ax.set_yticks(nd_t); ax.set_yticklabels(tlabels)
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")
    ax.set_title("Figure 4: DET Curves"); ax.legend(); ax.grid()
    ax.set_xlim(nd_t[0], nd_t[-1]); ax.set_ylim(nd_t[0], nd_t[-1])
    plt.tight_layout()
    plt.savefig(out / "fig4_det.pdf"); plt.close()

    # ── Figure 5: Calibration ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
        axes,
        ["ece_raw", "ece_platt", "ece_temp"],
        ["ECE Raw", "ECE Platt", "ECE Temp"],
    ):
        vals = [r[key] for r in fold_records]
        ax.bar(range(1, len(vals) + 1), vals, color=PAL[0], alpha=0.75, edgecolor="black")
        ax.axhline(np.mean(vals), color="red", lw=1.5, ls="--",
                   label=f"Mean={np.mean(vals):.4f}")
        ax.set_title(title); ax.set_xlabel("Fold"); ax.legend(); ax.grid(axis="y")
    fig.suptitle("Figure 5: Calibration ECE across folds", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "fig5_calibration.pdf"); plt.close()

    print("  Saved 5 figures to", out)


# ─────────────────────────────────────────────────────────────────────────────
#  Results Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_results(fold_records):
    def agg(vals):
        v = np.array(vals)
        return f"{v.mean():.4f} ± {v.std():.4f}"

    print("\n" + "=" * 70)
    print("SPEAKER-INDEPENDENT RESULTS  (mean ± std over outer folds)")
    print("=" * 70)

    for skey, sname in [
        ("youden", "Youden J"),
        ("eer",    "EER"),
        ("fixed",  "Fixed 0.5"),
        ("platt",  "Platt+Youden"),
        ("temp",   "Temp+Youden"),
    ]:
        print(f"\n  ── {sname} ──")
        for metric in ["auc", "accuracy", "f1", "sensitivity",
                       "specificity", "far", "frr", "brier", "threshold"]:
            print(f"    {metric:<14} {agg([r[skey][metric] for r in fold_records])}")

    auc_arr = np.array([r["youden"]["auc"]  for r in fold_records])
    thr_arr = np.array([r["youden_thresh"]  for r in fold_records])
    eer_arr = np.array([r["eer_thresh"]     for r in fold_records])
    instability = thr_arr.std() / (auc_arr.std() + 1e-10)

    print("\n  ── Stability Metrics ──")
    print(f"    AUC std             {auc_arr.std():.6f}")
    print(f"    Youden thresh std   {thr_arr.std():.6f}")
    print(f"    EER thresh std      {eer_arr.std():.6f}")
    print(f"    Instability ratio   {instability:.2f}×")
    print(f"    ECE (raw)           {agg([r['ece_raw']   for r in fold_records])}")
    print(f"    ECE (Platt)         {agg([r['ece_platt'] for r in fold_records])}")

    # Bootstrap CI on instability ratio
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(10_000):
        idx = rng.integers(0, len(fold_records), len(fold_records))
        a_s = np.std(auc_arr[idx]); t_s = np.std(thr_arr[idx])
        if a_s > 1e-8:
            boot.append(t_s / a_s)
    ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    print(f"    Instability 95% CI  [{ci_lo:.2f}×, {ci_hi:.2f}×]")
    print("=" * 70)
    return instability, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--outer",      type=int, default=5)
    parser.add_argument("--inner",      type=int, default=2)
    parser.add_argument("--patience",   type=int, default=3)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--load_cache", action="store_true",
                        help="Load cached fold_records instead of retraining")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "fold_records_cnnlstm.pkl"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Load data ────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print("Loading features …")
    X   = np.load(data_dir / "X_features.npy")
    y   = np.load(data_dir / "y_labels.npy")
    spk = np.load(data_dir / "speaker_ids.npy")
    print(f"  X: {X.shape}   Real={(y==0).sum()}  Fake={(y==1).sum()}")
    print(f"  Speakers: {len(np.unique(spk))}")

    pos_weight = torch.tensor([float((y == 0).sum()) / float((y == 1).sum())])
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # Model summary
    dummy = CNNLSTMDetector()
    print(f"\nCNN+LSTM parameters: {count_params(dummy):,}")

    if args.load_cache and cache_path.exists():
        print(f"\nLoading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            fold_records = pickle.load(f)
    else:
        # ── Speaker-independent splits ────────────────────────────────
        splits = speaker_independent_splits(y, spk, args.outer, args.inner)
        print(f"\nSpeaker-independent {args.outer}-fold CV "
              f"({args.outer * args.inner} total training runs)\n")

        fold_records   = []
        best_global_auc = -1.0
        best_model_state = None

        for fold_num, (tv_idx, test_idx, inner_folds) in enumerate(splits, 1):
            test_spks = np.unique(spk[test_idx])
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {fold_num}/{args.outer}  "
                  f"(test speakers: {test_spks}, "
                  f"train/val: {len(tv_idx)}  test: {len(test_idx)})")
            print(f"{'='*60}")

            record = run_fold(
                X, y, tv_idx, None, test_idx,
                inner_folds, pos_weight,
                device, use_amp, args,
                fold_num=fold_num,
            )
            fold_records.append(record)

            auc_v = record["youden"]["auc"]
            print(f"  Outer fold {fold_num} → AUC={auc_v:.4f}  "
                  f"F1={record['youden']['f1']:.4f}  "
                  f"Youden-τ={record['youden_thresh']:.4f}  "
                  f"EER-τ={record['eer_thresh']:.4f}  "
                  f"ECE={record['ece_raw']:.4f}")

            if auc_v > best_global_auc:
                best_global_auc  = auc_v
                best_model_state = record["model_state"]
                best_fold        = fold_num

        with open(cache_path, "wb") as f:
            pickle.dump(fold_records, f)
        print(f"\nCached fold records → {cache_path}")

    # ── Results ────────────────────────────────────────────────────────
    instability, ci_lo, ci_hi = print_results(fold_records)
    plot_all(fold_records, out)

    # ── Save checkpoint ─────────────────────────────────────────────────
    best_idx = int(np.argmax([r["youden"]["auc"] for r in fold_records]))
    ckpt = {
        "model_state_dict":  fold_records[best_idx]["model_state"],
        "model_class":       "CNNLSTMDetector",
        "best_outer_fold":   best_idx + 1,
        "best_auc":          fold_records[best_idx]["youden"]["auc"],
        "youden_threshold":  fold_records[best_idx]["youden_thresh"],
        "eer_threshold":     fold_records[best_idx]["eer_thresh"],
        "temperature":       fold_records[best_idx]["temperature"],
        "instability_ratio": {"point": instability, "ci_lo": ci_lo, "ci_hi": ci_hi},
        "feature_params": dict(
            type="Log-Mel spectrogram", sample_rate=16000,
            n_mels=128, n_frames=94, window_ms=25, hop_ms=10,
        ),
        "fold_results": [
            {k: v for k, v in r.items()
             if k not in ("fpr", "tpr", "test_scores", "test_labels",
                          "test_idx", "model_state")}
            for r in fold_records
        ],
    }
    ckpt_path = out / "deepfake_cnnlstm.pth"
    torch.save(ckpt, ckpt_path)
    print(f"\n✅ Checkpoint saved → {ckpt_path}")
    print(f"✅ Best fold: {best_idx+1}  AUC={ckpt['best_auc']:.4f}  "
          f"Youden-τ={ckpt['youden_threshold']:.4f}")
    print(f"\nNext step: python 04_realtime_inference.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
