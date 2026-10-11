"""
Stage1 opt-in 후보 학습 스크립트: MViTv2-S(Kinetics-400 pretrained) + AIHub597
교통사고 영상 데이터 + timestamp-phase 기반 synthetic recapture augmentation.

절대 기존 배포 파이프라인(model/stage1/best.pt, train_stage1.py,
stage1_sequence_features.py)을 건드리지 않는다 - 이 스크립트는 완전히 별도의
체크포인트(기본: model/stage1/mvit_aihub_candidate.pt)를 만드는 순수 실험용
opt-in 후보다. inference.py/build_submit.py는 이 체크포인트를 참조하지 않는다.

사용 예:
    python train_stage1_mvit_aihub.py \
        --manifest_dir /data/aihub_manifest \
        --extract_dir  /data/aihub_extracted \
        --save_dir     /workspace/model/stage1 \
        --epochs 5 --batch 2
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from stage1_aihub_dataset import AIHubStage1MViTDataset
from stage1_mvit_model import Stage1MViT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Stage1 MViT + AIHub597 opt-in candidate training")

    p.add_argument("--manifest_dir", type=str, default=r"D:\workspace_mj\aihub597_stage1_manifest")
    p.add_argument("--extract_dir", type=str, default=r"D:\workspace_mj\aihub597_stage1_extracted")
    p.add_argument("--save_dir", type=str, default="model/stage1")
    p.add_argument("--checkpoint_name", type=str, default="mvit_aihub_candidate.pt")

    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--recapture_size", type=int, default=320)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=2, help="source-video 레벨 batch. 실제 MViT batch = batch*2")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--train_source_limit", type=int, default=None, help="manifest가 이미 제한된 크기라면 보통 불필요")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(args.manifest_dir)
    extract_dir = Path(args.extract_dir)

    train_manifest = manifest_dir / "train.csv"
    val_manifest = manifest_dir / "val.csv"

    train_dataset = AIHubStage1MViTDataset(
        root=extract_dir, manifest=train_manifest,
        frames=args.frames, size=args.size, recapture_size=args.recapture_size,
        train=True,
    )
    val_dataset = AIHubStage1MViTDataset(
        root=extract_dir, manifest=val_manifest,
        frames=args.frames, size=args.size, recapture_size=args.recapture_size,
        train=False,
    )

    if args.train_source_limit is not None and len(train_dataset) > args.train_source_limit:
        g = torch.Generator()
        g.manual_seed(args.seed)
        selected = torch.randperm(len(train_dataset), generator=g)[: args.train_source_limit].tolist()
        train_dataset = Subset(train_dataset, selected)

    print("=== Stage1 MViT + AIHub597 opt-in candidate ===")
    print(f"train source videos: {len(train_dataset)} (train samples: {len(train_dataset) * 2})")
    print(f"val source videos:   {len(val_dataset)} (val samples: {len(val_dataset) * 2})")
    print(f"source batch size: {args.batch} -> effective MViT batch: {args.batch * 2}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = Stage1MViT(pretrained=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    best_f1 = -1.0
    checkpoint_path = save_dir / args.checkpoint_name

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        nan_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Train")
        for x, y in pbar:
            x = x.flatten(0, 1).to(device, non_blocking=True)
            y = y.flatten().to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)

            if not torch.isfinite(loss):
                nan_batches += 1
                print(f"[WARN] non-finite loss at epoch {epoch + 1}, skipping batch (count={nan_batches})")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            pbar.set_postfix(loss=running_loss / max(1, n_seen))

        avg_loss = running_loss / max(1, n_seen)

        if nan_batches > 0:
            print(f"[WARN] epoch {epoch + 1}: {nan_batches} non-finite batches skipped")

        # ---------------- validation ----------------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Val"):
                x = x.flatten(0, 1).to(device, non_blocking=True)
                y = y.flatten()

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                    logits = model(x)

                pred = logits.argmax(dim=1)
                y_true.extend(y.tolist())
                y_pred.extend(pred.cpu().tolist())

        macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
        acc = float(np.mean([int(a == b) for a, b in zip(y_true, y_pred)])) if y_true else 0.0

        print(f"True: {Counter(y_true)}")
        print(f"Pred: {Counter(y_pred)}")
        print("Confusion Matrix:")
        if y_true:
            print(confusion_matrix(y_true, y_pred))

        print(
            f"[Stage1-MViT-AIHub] Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | val_acc={acc:.4f} | val_macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            torch.save(
                {
                    "model": model.net.state_dict(),
                    "size": args.size,
                    "frames": args.frames,
                    "recapture_size": args.recapture_size,
                    "val_macro_f1": macro_f1,
                    "val_acc": acc,
                    "epoch": epoch + 1,
                    "source": "aihub597 + synthetic recapture (MelonChicken-style port)",
                },
                checkpoint_path,
            )
            print(f"[Stage1-MViT-AIHub] best checkpoint updated: {best_f1:.4f} -> {checkpoint_path}")

    print()
    print(f"[Stage1-MViT-AIHub] training done. best val_macro_f1={best_f1:.4f}")
    print(f"[Stage1-MViT-AIHub] checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
