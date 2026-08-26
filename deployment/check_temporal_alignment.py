from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check where the clip/key frame lies inside the 9-frame B8 temporal window."
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--pkl", required=True)
    args = p.parse_args()

    repo = Path(args.repo)
    pkl_path = Path(args.pkl)

    sys.path.insert(0, str(repo / "data"))
    import boxinfo  # noqa: F401

    with pkl_path.open("rb") as f:
        annotations = pickle.load(f)

    offset_patterns = Counter()

    for video_id, clips in annotations.items():
        for clip_id, clip in clips.items():
            key = int(clip_id)
            frames = sorted(int(x) for x in clip["frame_boxes_dct"].keys())
            offsets = tuple(frame - key for frame in frames)
            offset_patterns[offsets] += 1

    print("Temporal offset patterns relative to clip/key frame:")
    for pattern, count in offset_patterns.most_common(20):
        print(f"{count:>5} clips: {pattern}")


if __name__ == "__main__":
    main()
