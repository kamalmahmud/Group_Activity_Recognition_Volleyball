import os
import time

import torch
from tqdm import tqdm

from .evaluator import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    device_type = device.type if isinstance(device, torch.device) else str(device)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        if len(batch) == 2:
            frames, labels = batch
            mask = None
        elif len(batch) == 3:
            frames, labels, mask = batch

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda", ):
            if mask is not None:
                outputs = model(frames, mask=mask)
            else:
                outputs = model(frames)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * frames.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

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
        scheduler=None,
        num_epochs: int = 30,
        save_dir: str = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0
    best_epoch = -1
    history = []

    print(f"\nStarting training for {num_epochs} epochs …\n{'─' * 60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            class_names=class_names,
            print_report=False,
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:>3}/{num_epochs}] "
            f"| train loss: {train_loss:.4f}  acc: {train_acc:.4f} "
            f"| val loss: {val_loss:.4f}  acc: {val_acc:.4f} "
            f"| lr: {current_lr:.2e} "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            ckpt_path = os.path.join(save_dir, "best_model.pth")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)

            print(f"  New best model saved (val_acc={val_acc:.4f})")

    print(f"\n{'─' * 60}")
    print(f"Training complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}.")

    print("\nLoading best model for test-set evaluation …")

    checkpoint = torch.load(
        os.path.join(save_dir, "best_model.pth"),
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n" + "═" * 60)
    print("TEST SET EVALUATION")
    print("═" * 60)

    evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names=class_names,
        print_report=True,
    )

    return model, history
