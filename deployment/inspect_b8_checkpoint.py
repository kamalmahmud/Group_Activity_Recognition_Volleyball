from __future__ import annotations

import argparse
import torch


def strip_common_wrappers(state):
    state = dict(state)
    prefixes = ("module.", "_orig_mod.", "model.")
    changed = True
    while changed and state:
        changed = False
        for prefix in prefixes:
            if all(k.startswith(prefix) for k in state):
                state = {k[len(prefix):]: v for k, v in state.items()}
                changed = True
                break
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    print("Checkpoint top-level type:", type(ckpt).__name__)

    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if not isinstance(state, dict):
        print("State payload is not a dict:", type(state).__name__)
        return

    state = strip_common_wrappers(state)

    print("\nState tensors:", len(state))
    print("\nFirst 25 keys/shapes:")
    for k in list(state.keys())[:25]:
        v = state[k]
        print(f"{k:55s} {tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")

    keys = [
        "player_lstm.weight_ih_l0",
        "player_lstm.weight_hh_l0",
        "frame_lstm.weight_ih_l0",
        "frame_lstm.weight_hh_l0",
        "classifier.0.weight",
        "classifier.1.weight",
        "classifier.4.weight",
    ]

    print("\nB8 architecture-defining tensors:")
    for k in keys:
        if k in state:
            print(f"{k:40s} {tuple(state[k].shape)}")
        else:
            print(f"{k:40s} MISSING")


if __name__ == "__main__":
    main()
