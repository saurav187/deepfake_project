import json
import os
from collections import Counter

import numpy as np
import torch
from torch import nn

from .audio import channel, decode, prepare, windows
from .data import digest, entries, payload
from .metrics import slices
from .models import Student, Teacher, tensor


def logits(model, x, student, device):
    return torch.stack([model(tensor(w, student, device)) for _, w in windows(x)])


def score(model, student, rows, local, device):
    out = []
    model.eval()
    with torch.inference_mode():
        for r in rows:
            x, _ = prepare(decode(payload(local, r)))
            if list(windows(x)):
                s = logits(model, x, student, device).cpu().numpy()
                out.append((dict(r, length="<4s" if len(x) < 64000 else "4s+"), s))
    return out


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg, student):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "student" if student else "teacher"
    local, work = cfg.local, cfg.data / "runs"
    work.mkdir(parents=True, exist_ok=True)
    rows = entries(local, {"train"})
    dev = entries(local, {"select"})
    model = (Student() if student else Teacher(cfg.encoder)).to(device)
    ref = None
    teacher_path = work / "teacher.best.pt"
    if student:
        ref = Teacher(cfg.encoder).to(device)
        ref.load_state_dict(
            torch.load(teacher_path, map_location=device, weights_only=True)
        )
        ref.eval()
    provenance = [
        digest(local / "index.json"),
        str(cfg.encoder),
        cfg.seed,
        cfg.steps,
        digest(teacher_path) if student else "teacher",
    ]
    last = work / f"{name}.last.pt"
    best_path = work / f"{name}.best.pt"
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4 if student else 1e-5
    )
    begin, best, stale = 0, float("inf"), 0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["provenance"] != provenance:
            raise ValueError("Checkpoint inputs changed. Choose a separate DATA run.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        begin, best, stale = state["epoch"] + 1, state["best"], state["stale"]
    cells = [(r["pair"], r["language"], r["label"]) for r in rows]
    counts = Counter(cells)
    weights = np.array([1 / counts[c] for c in cells])
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(begin, cfg.epochs):
        if stale >= 3:
            break
        rng = np.random.default_rng(cfg.seed + epoch)
        torch.manual_seed(cfg.seed + epoch)
        model.train()
        for idx in rng.choice(len(rows), cfg.steps, p=weights / weights.sum()):
            r = rows[idx]
            raw = decode(payload(local, r))
            name_aug = rng.choice(
                ["clean"] * 6
                + [
                    "opus",
                    "mp3",
                    "aac",
                    "phone",
                    "rate",
                    "playback",
                    "noise",
                    "rawboost",
                ]
            )
            x, _ = prepare(channel(raw, name_aug, rng))
            chunks = list(windows(x))
            if not chunks:
                continue
            _, w = chunks[rng.integers(len(chunks))]
            pred = model(tensor(w, student, device))
            target = torch.tensor(float(r["label"]), device=device)
            loss = loss_fn(pred, target)
            if ref is not None:
                with torch.no_grad():
                    soft = torch.sigmoid(ref(tensor(w, False, device)) / 2)
                loss = 0.5 * loss + 2 * loss_fn(pred / 2, soft)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        results = score(model, student, dev, local, device)
        groups = slices([r for r, _ in results], [s.mean() for _, s in results])
        if (
            not (metric := groups.get("domain/unseen", {})).get("n")
            or metric["eer"] is None
        ):
            raise ValueError("Need matched held-family selection data")
        eer = metric["eer"]
        if eer < best:
            best, stale = eer, 0
            atomic_save(model.state_dict(), best_path)
        else:
            stale += 1
        atomic_save(
            {
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "provenance": provenance,
            },
            last,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "unseen_eer": eer,
                    "scored": len(results),
                    "total": len(dev),
                }
            ),
            flush=True,
        )
