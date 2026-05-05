import pickle
from typing import Callable, Dict, Optional, Tuple

from torch.utils.data import Dataset

from ._builders import DatasetIndexBuildersMixin
from ._getters import DatasetGettersMixin
from ._helpers import DatasetHelpersMixin
from .constants import SPLITS


class VolleyballDataset(
    DatasetGettersMixin,
    DatasetIndexBuildersMixin,
    DatasetHelpersMixin,
    Dataset,
):
    def __init__(
            self,
            pkl_path: str,
            videos_path: str,
            frame_transform: Optional[Callable] = None,
            crop_transform: Optional[Callable] = None,
            split: str = "train",
            mode: str = "frame",
            player_order: str = "player_id",
            include_lost: bool = True,
            include_generated: bool = True,
    ):
        self.videos_path = videos_path
        self.frame_transform = frame_transform
        self.crop_transform = crop_transform
        self.pkl_path = pkl_path
        self.split = split
        self.mode = mode
        self.player_order = player_order
        self.include_lost = include_lost
        self.include_generated = include_generated

        self._mode_dispatch: Dict[str, Tuple[Callable, Callable]] = {
            "frame": (
                self._get_frame,
                self._build_frame_index,
            ),
            "person": (
                self._get_person,
                self._build_person_index,
            ),
            "frame_person": (
                self._get_frame_person,
                self._build_frame_person_index,
            ),
            "temporal_frame": (
                self._get_temporal_frame,
                self._build_temporal_clip_index,
            ),
            "temporal_person": (
                self._get_temporal_person,
                self._build_temporal_person_index,
            ),
            "temporal_person_clip": (
                self._get_temporal_person_clip,
                self._build_temporal_person_clip_index,
            ),
        }

        if mode not in self._mode_dispatch:
            raise ValueError(
                f"mode must be one of {list(self._mode_dispatch)}, got {mode!r}"
            )

        if split not in SPLITS:
            raise ValueError(
                f"split must be one of {list(SPLITS)}, got {split!r}"
            )

        with open(self.pkl_path, "rb") as file:
            self.annotations = pickle.load(file)

        _, build_index = self._mode_dispatch[mode]
        self.samples = build_index(SPLITS[split])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        get_item, _ = self._mode_dispatch[self.mode]
        return get_item(self.samples[idx])
