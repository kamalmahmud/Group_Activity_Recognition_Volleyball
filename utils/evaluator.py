import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, )
from tqdm import tqdm

matplotlib.use("Agg")


def evaluate(model, loader, criterion, device, class_names, print_report=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    device_type = device.type if isinstance(device, torch.device) else str(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if len(batch) == 2:
                frames, labels = batch
                mask = None
            elif len(batch) == 3:
                frames, labels, mask = batch

            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device_type == "cuda"):
                if mask is not None:
                    logits = model(frames, mask=mask)
                else:
                    logits = model(frames)
                loss = criterion(logits, labels)

            running_loss += loss.item() * frames.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = running_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    if print_report:
        print_classification_report(all_labels, all_preds, class_names)

    return mean_loss, accuracy, all_preds, all_labels


# ── Classification report ─────────────────────────────────────────────────────

def print_classification_report(labels, preds, class_names):
    """Pretty-print sklearn classification report."""
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=2,
    )
    print("\nClassification Report")
    print("─" * 60)
    print(report)


# ── Percentile confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(
        labels,
        preds,
        class_names,
        save_path="confusion_matrix.png",
        figsize=(10, 8),
        cmap="Blues",
):
    """
    Plot and save_path a row-normalised (percentile) confusion matrix.

    Each cell shows the percentage of true-class samples predicted
    as a given class, so every row sums to 100 %.
    """
    cm = confusion_matrix(labels, preds)
    # Row-normalise → percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    n = len(class_names)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)

    # Colour-bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    # Axis ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate every cell
    thresh = 50.0  # flip text colour above this percentage
    for i in range(n):
        for j in range(n):
            pct = cm_pct[i, j]
            count = cm[i, j]
            color = "white" if pct > thresh else "black"
            ax.text(
                j, i,
                f"{pct:.1f}%\n({count})",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised %)", fontsize=14, pad=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def full_evaluation(
        model,
        loader,
        criterion,
        device,
        class_names,
        cm_save_path="confusion_matrix.png",
):
    """
    Run evaluate(), print the classification report, and save_path the
    confusion matrix in one call.  Intended for final test-set scoring.
    """
    loss, acc, preds, labels = evaluate(
        model, loader, criterion, device,
        class_names=class_names, print_report=True
    )
    plot_confusion_matrix(
        labels, preds,
        class_names=class_names,
        save_path=cm_save_path,
    )
    return loss, acc, preds, labels
