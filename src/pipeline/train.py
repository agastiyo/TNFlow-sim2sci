"""Train TNFlow: combined composition + grain NLL.

Noised run: training batches and the val/ood eval splits are noise-augmented (dataloader.add_noise).
One checkpoint, selected on the VALIDATION split only (never OOD):
  tnflow_checkpoint.pt  -- best (lowest) validation composition NLL

Per-epoch train / val / OOD curves (including val top-N) go to config.HISTORY_PATH, rewritten
each epoch, so top-N can still be read off the training curve without a dedicated checkpoint.
Run from the project root:  python3.10 -m src.pipeline.train
"""

import json

import numpy as np
import torch

from src import config
from src.data_utils.dataloader import DataLoader
from src.data_utils.store import mixture_id
from src.model.cnf import free_to_full_simplex
from src.model.tnflow import TNFlow
from src.pipeline.checkpoint import save_best
from src.pipeline.metrics import detection_score, grain_log_error, ranks_from_scores, recall_at_k

EPOCHS = 1000
BATCHES = 512
GRAD_CLIP = 5.0
LR = 1e-3
GRAIN_WEIGHT = 1.0  # weight on the (masked) grain NLL
SEED = 10025734     # weight init + dropout (DataLoader shuffle/noise is seeded separately)

TRAIN_EVAL_N = 0  # fixed clean subsample of train scored each epoch for the train curve

MODEL = dict(transformer_size="medium", comp_flow_size="medium", grain_flow_size="small",
             tokenizer="norm")

ACC_TOPK = 5
ACC_SAMPLES = 64

SPLITS = ("train", "val", "ood")
METRICS = ("loss", "topN", "top5", "grain_dex", "grain_within2")


def evaluate(model, device, chunks, tag):
    """One set's NLL, top-N/top-k recall, and grain error.

    ``grain_dex`` is teacher-forced (conditioned on the true composition) -- a training
    diagnostic, not an achievable inference-time number.

    Args:    model: TNFlow; device; chunks: (spectra, lams, targets, grains) batch-tensor lists;
             tag: str label for the progress line.
    Returns: dict with keys loss, topN, top5, grain_dex, grain_within2.
    """
    spectra, lams, targets, grains = chunks
    model.eval()
    vl, vn, correct_N, correct_k, total = 0.0, 0, 0, 0, 0
    dlogs = []
    with torch.no_grad():
        for i, (specs, lam, full_tgt, grn) in enumerate(zip(spectra, lams, targets, grains), start=1):
            specs, lam = specs.to(device), lam.to(device)
            full_tgt, grn = full_tgt.to(device), grn.to(device)
            tgt = full_tgt[:, :-1]

            ctx = model.encode(lam, specs)
            dist = model.cnf.dist(ctx)
            vl += -dist.log_prob(model.cnf.smooth(tgt)).mean().item() * len(specs); vn += len(specs)

            draws = free_to_full_simplex(dist.sample((ACC_SAMPLES,)))   # [ACC_SAMPLES, B, K]
            present = full_tgt > 0
            n_present = present.sum(dim=1, keepdim=True)
            ranks = ranks_from_scores(detection_score(draws))
            correct_N += recall_at_k(ranks, present, n_present)
            correct_k += recall_at_k(ranks, present, ACC_TOPK)
            total += len(specs)

            gmask = present & torch.isfinite(grn)
            if gmask.any():
                gpred = model.grain.predict_median(ctx, full_tgt, n=ACC_SAMPLES)
                dlogs.append(grain_log_error(gpred, grn)[gmask].cpu())
            print(f"  {tag} batch {i}/{len(spectra)} | loss {vl/vn:7.3f} | "
                  f"top-N {correct_N/total:5.3f} | top-{ACC_TOPK} {correct_k/total:5.3f}", end='\r')
    nan = float("nan")
    if dlogs:
        d = torch.cat(dlogs)
        g_med, g_w2 = d.median().item(), (d < np.log10(2.0)).float().mean().item()
    else:
        g_med = g_w2 = nan
    return {
        "loss": vl / vn if vn else nan,
        "topN": correct_N / total if total else nan,
        "top5": correct_k / total if total else nan,
        "grain_dex": g_med,
        "grain_within2": g_w2,
    }


def _finite(x):
    """NaN/inf -> None so the history file stays strict-JSON readable."""
    return None if x is None or not np.isfinite(x) else float(x)


def _save_history(history, meta):
    """Write the per-epoch curves to config.HISTORY_PATH (temp file + replace)."""
    path = config.HISTORY_PATH
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, **history}, indent=1))
    tmp.replace(path)


def main():
    torch.manual_seed(SEED)
    dl = DataLoader(batches=BATCHES)

    # Fixed evaluation sets, batched once. ``train`` stays CLEAN (invertibility-ceiling diagnostic);
    # ``val``/``ood`` are noised ONCE here -- one frozen noise realization scored every epoch, matching
    # the single noisy draw a real observation gives and test.py's noised eval. val loss (noised)
    # drives selection. All three sets stay identical every epoch so the curves are comparable.
    sub = np.random.default_rng(SEED).choice(
        len(dl.train), min(TRAIN_EVAL_N, len(dl.train)), replace=False)
    eval_sets = {
        "train": dl._batch([dl.train[i] for i in sub]),
        "val": dl._batch(dl.test, noise=True),
        "ood": dl._batch(dl.ood, noise=True),
    }
    held_out = sorted({mixture_id(s.target) for s in dl.ood})
    print(f"  splits: train {len(dl.train)} | val {len(dl.test)} | ood {len(dl.ood)} "
          f"({len(held_out)} held-out mixtures)")
    print(f"  NOISED train batches + val/ood eval; "
          f"train curve on a fixed {len(sub)}-sample CLEAN subset of train (diagnostic)")

    # Grain standardization: log10(grain/µm) mean & std over the training set's present grains.
    _lg = np.log10(np.concatenate([s.grains[(s.target > 0) & np.isfinite(s.grains)] for s in dl.train]))
    model_cfg = dict(MODEL, log_grain_mean=float(_lg.mean()), log_grain_std=float(_lg.std()))

    device = config.get_device()
    model = TNFlow(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_loss = float("inf")       # validation only -- OOD never enters selection
    best_loss_epoch = None

    history = {"epochs": [], "train_objective": [],
               **{s: {m: [] for m in METRICS} for s in SPLITS}}
    meta = {
        "model_config": model_cfg,
        "hyperparameters": {"epochs": EPOCHS, "batches": BATCHES, "grad_clip": GRAD_CLIP,
                            "lr": LR, "grain_weight": GRAIN_WEIGHT, "seed": SEED,
                            "acc_topk": ACC_TOPK, "acc_samples": ACC_SAMPLES,
                            "train_eval_n": int(len(sub))},
        "split_sizes": {"train": len(dl.train), "val": len(dl.test), "ood": len(dl.ood)},
        "held_out_mixtures": held_out,
        "notes": ("train.* curves are a fixed clean subsample of the training split; "
                  "train_objective is the optimized comp+grain NLL on noised batches. "
                  "val/ood eval sets are noised. OOD is logged only -- never used for checkpoint selection."),
    }

    print("Training Start!")
    for epoch in range(EPOCHS):
        train_spectra, train_lams, train_targets, train_grains = dl._batch(
            dl.train, shuffle=True, noise=True)  # fresh shuffle + noise augmentation this epoch

        model.train()
        running, n = 0.0, 0
        nb = len(train_spectra)
        for b, (specs, lam, full_tgt, grn) in enumerate(
                zip(train_spectra, train_lams, train_targets, train_grains), start=1):
            specs, lam = specs.to(device), lam.to(device)
            full_tgt, grn = full_tgt.to(device), grn.to(device)
            c = model.encode(lam, specs)
            comp_nll = -model.cnf.log_prob(c, full_tgt[:, :-1]).mean()
            mask = (full_tgt > 0) & torch.isfinite(grn)
            g_sum, g_n = model.grain.log_prob(c, full_tgt, grn, mask)
            loss = comp_nll + GRAIN_WEIGHT * (-g_sum / max(g_n, 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(specs); n += len(specs)
            print(f"Epoch {epoch+1}/{EPOCHS} | train batch {b}/{nb} | avg loss {running/n:7.3f}", end='\r')
        train_objective = running / n

        scored = {s: evaluate(model, device, eval_sets[s], s) for s in SPLITS}
        scheduler.step(scored["val"]["loss"])

        history["epochs"].append(epoch + 1)
        history["train_objective"].append(_finite(train_objective))
        for s in SPLITS:
            for m in METRICS:
                history[s][m].append(_finite(scored[s][m]))

        v, o = scored["val"], scored["ood"]
        print(f"Epoch {epoch+1}/{EPOCHS} | obj {train_objective:7.3f} | "
              f"val loss {v['loss']:7.3f} topN {v['topN']:5.3f} top{ACC_TOPK} {v['top5']:5.3f} "
              f"gΔ {v['grain_dex']:4.2f} | ood loss {o['loss']:7.3f} topN {o['topN']:5.3f} "
              f"top{ACC_TOPK} {o['top5']:5.3f} gΔ {o['grain_dex']:4.2f}" + " " * 4)

        def _metrics(selected_by):
            return {"selected_by": selected_by, "train_objective": train_objective,
                    **{f"{s}_{m}": scored[s][m] for s in SPLITS for m in METRICS},
                    "hyperparameters": meta["hyperparameters"]}

        if v["loss"] < best_loss:                       # selection: validation loss
            best_loss, best_loss_epoch = v["loss"], epoch + 1
            save_best(config.BEST_LOSS_CHECKPOINT, model, optimizer, scheduler, epoch,
                      _metrics("val_loss"))
            print(f"  new best val loss ({best_loss:.3f}) -> {config.BEST_LOSS_CHECKPOINT.name}")

        meta["best"] = {"val_loss": {"epoch": best_loss_epoch, "value": _finite(best_loss)}}
        _save_history(history, meta)

    print(f"Training complete!  best val loss {best_loss:.3f} (epoch {best_loss_epoch})")
    print(f"  history -> {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
