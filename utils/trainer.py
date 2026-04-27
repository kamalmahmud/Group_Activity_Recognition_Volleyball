import os
import time

import torch
from tqdm import tqdm

from .evaluator import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        # VolleyballDataset may return (frames, labels) or (frames, crops, labels)
        # depending on the mode — handle both cases gracefully.
        if len(batch) == 3:
            frames, _, labels = batch
        else:
            frames, labels = batch

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        class_names,
        num_epochs: int = 30,
        save_dir: str = "checkpoints",

):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = -1
    history = []

    print(f"\nStarting training for {num_epochs} epochs …\n{'─' * 60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device,
            class_names=class_names, print_report=False
        )

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:>3}/{num_epochs}] "
            f"| train loss: {train_loss:.4f}  acc: {train_acc:.4f} "
            f"| val loss: {val_loss:.4f}  acc: {val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    print(f"\n{'─' * 60}")
    print(f"Training complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}.")

    # ── Final evaluation on test set using best model ────────────────────────
    print("\nLoading best model for test-set evaluation …")
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n" + "═" * 60)
    print("TEST SET EVALUATION")
    print("═" * 60)
    evaluate(
        model, test_loader, criterion, device,
        class_names=class_names, print_report=True
    )

    return model, history
