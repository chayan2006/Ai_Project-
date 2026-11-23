"""CRNN training (report 5.3.2, Table 6, Appendix D.8).

Stages: A (public handwriting: IIIT-HW-Dev, IAM), B (SynJama register lines), C (HandFilled).
Each stage reads a packed line split (``recognise.data.build_lines``) and can start from the
previous stage's checkpoint. The best checkpoint by validation CER is kept, with a model card.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from ..layout.unet import device
from .charset import NUM_CLASSES
from .crnn import CRNN, greedy_decode, n_params
from .data import LineStore, augment, bucketed_batches, collate


def cer(refs: list[str], hyps: list[str]) -> float:
    dist = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    return dist / max(1, sum(len(r) for r in refs))


@torch.no_grad()
def evaluate(model: CRNN, store: LineStore, dev, limit: Optional[int] = 3000, batch_size: int = 32,
             seed: int = 0) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    refs, hyps, fields = [], [], []
    for k, idx in enumerate(bucketed_batches(store, batch_size, rng, shuffle=False, limit=limit)):
        if dev.type == "mps" and k % 50 == 0:
            torch.mps.empty_cache()                   # MPS caches memory per batch shape
        b = collate([store.image(int(i)) for i in idx], [store.lines[int(i)]["text"] for i in idx],
                    [store.lines[int(i)]["field"] for i in idx])
        out = greedy_decode(model(b.images.to(dev)), b.input_lengths)
        refs += b.texts
        hyps += [t for t, _ in out]
        fields += b.fields
    model.train()
    if dev.type == "mps":
        torch.mps.empty_cache()
    per_field = defaultdict(lambda: ([], []))
    for r, h, f in zip(refs, hyps, fields):
        per_field[f][0].append(r)
        per_field[f][1].append(h)
    return {"cer": cer(refs, hyps), "line_accuracy": float(np.mean([r == h for r, h in zip(refs, hyps)])),
            "lines": len(refs), "cer_by_field": {f: round(cer(*v), 4) for f, v in sorted(per_field.items())}}


def load_checkpoint(path: str | Path) -> CRNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CRNN(num_classes=ckpt.get("num_classes", NUM_CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def train(stage: str, data: str | Path, val: str | Path, out: str | Path, epochs: int = 15,
          init: Optional[str | Path] = None, batch_size: int = 32, lr: float = 1e-3, seed: int = 0,
          patience: int = 5, max_minutes: Optional[float] = None, train_limit: Optional[int] = None,
          val_limit: int = 3000, log: Callable[[str], None] = print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    tr, va = LineStore(data), LineStore(val)
    model = load_checkpoint(init) if init else CRNN()
    model = model.to(dev).train()
    steps_per_epoch = len(bucketed_batches(tr, batch_size, np.random.default_rng(0), limit=train_limit))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = dev.type == "cuda"                      # Table 6: mixed precision (FP16) on GPU, FP32 elsewhere
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"stage {stage}: {len(tr)} train / {len(va)} val lines, {n_params(model):,} params, device {dev}, "
        f"{steps_per_epoch} steps per epoch")
    t0, best, stale, history = time.time(), float("inf"), 0, []
    stopped = "completed"
    for epoch in range(1, epochs + 1):
        losses = []
        for step, idx in enumerate(bucketed_batches(tr, batch_size, rng, limit=train_limit), 1):
            imgs = [augment(tr.image(int(i)), rng) for i in idx]
            b = collate(imgs, [tr.lines[int(i)]["text"] for i in idx], [tr.lines[int(i)]["field"] for i in idx])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                log_probs = model(b.images.to(dev, non_blocking=True))
            log_probs = log_probs.float()
            if dev.type == "mps":                     # CTC is not implemented on Apple MPS: compute it on CPU
                loss = ctc(log_probs.cpu(), b.targets, b.input_lengths, b.target_lengths)
            else:
                loss = ctc(log_probs, b.targets.to(dev), b.input_lengths.to(dev), b.target_lengths.to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            if dev.type == "mps" and step % 200 == 0:
                torch.mps.empty_cache()
            if step % 500 == 0:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses[-500:]):.3f} "
                    f"({(time.time() - t0) / 60:.1f} min)")
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                stopped = f"time budget of {max_minutes} min reached in epoch {epoch}"
                break
        ev = evaluate(model, va, dev, limit=val_limit)
        history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "val_cer": round(ev["cer"], 4),
                        "val_line_acc": round(ev["line_accuracy"], 4), "minutes": round((time.time() - t0) / 60, 1)})
        log(f"epoch {epoch}: loss {np.mean(losses):.3f}  val CER {ev['cer']:.4f}  "
            f"line acc {ev['line_accuracy']:.3f}  ({(time.time() - t0) / 60:.1f} min)")
        if ev["cer"] < best:
            best, stale = ev["cer"], 0
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "num_classes": NUM_CLASSES, "stage": stage, "epoch": epoch}, out)
        else:
            stale += 1
        if stale >= patience:
            stopped = f"early stop: no val CER gain for {patience} epochs"
            break
        if stopped != "completed":
            break
    final = evaluate(load_checkpoint(out).to(dev), va, dev, limit=None)
    card = {"model": "CRNN (6 conv + BN, 2 x BiLSTM 256, CTC)", "params": n_params(model), "stage": stage,
            "init": str(init) if init else None, "train_data": str(data), "val_data": str(val),
            "train_lines": len(tr) if train_limit is None else train_limit, "epochs_planned": epochs,
            "batch_size": batch_size, "lr": lr, "seed": seed, "device": str(dev), "stopped": stopped,
            "history": history, "best_val_cer_subset": best, "val_full": final,
            "minutes": round((time.time() - t0) / 60, 1),
            "licence": "trained on synthetic SynJama pages only" if stage == "B" and not init else "see init model"}
    out.with_suffix(".card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    log(f"saved {out}: val CER {final['cer']:.4f}, line accuracy {final['line_accuracy']:.3f} ({stopped})")
    return card
