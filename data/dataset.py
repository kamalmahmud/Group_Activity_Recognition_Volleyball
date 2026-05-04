import os
import os.path
import pickle
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


SPLITS = {
    'train': ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
              "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54", ],
    'val': ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"],
    'test': ['4', '5', '9', '11', '14', '20', '21', '25', '29', '34', '35', '37', '43', '44', '45', '47']
}

GROUP_LABELS = {
            'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
            'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
        }

PLAYER_LABELS = {
            'blocking': 0, 'digging': 1, 'falling': 2, 'jumping': 3,
            'moving': 4, 'setting': 5, 'spiking': 6, 'standing': 7, 'waiting': 8
}


class VolleyballDataset(Dataset):
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
            "frame": (self._get_frame, self._build_frame_index),
            "person": (self._get_person, self._build_person_index),
            "frame_person": (self._get_frame_person, self._build_frame_person_index),
            "temporal_frame": (self._get_temporal_frame, self._build_temporal_clip_index),
            "temporal_person": (self._get_temporal_person, self._build_temporal_person_index),
        }
        if mode not in self._mode_dispatch:
            raise ValueError(f"mode must be one of {list(self._mode_dispatch)}, got {mode!r}")

        with open(self.pkl_path, "rb") as f:
            self.annotations = pickle.load(f)

        self.samples = self._mode_dispatch[mode][1](SPLITS[split])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        getter, _ = self._mode_dispatch[self.mode]
        return getter(self.samples[idx])

    # __getitem__ Methods-------------------------------------------------------------------------------
    def _get_frame(self, item):
        image = self._load_frame_image(item["path"])
        label = GROUP_LABELS[item["group_label_name"]]
        return image, label

    def _get_person(self, item: Dict[str, Any]) -> Dict[str, Any]:
        image = Image.open(item["path"]).convert("RGB")
        crop = image.crop(item["bbox"])
        crop = self._apply_crop_transform(crop)
        label = item["target"]
        return crop, label

    def _get_frame_person(self, item: Dict[str, Any]):
        image = Image.open(item["path"]).convert("RGB")

        W, H = image.size
        boxes = item["boxes"]

        left_boxes = []
        right_boxes = []

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2

            if center_x < W / 2:
                left_boxes.append(bbox)
            else:
                right_boxes.append(bbox)

        # keep max 6 persons per side
        left_boxes = left_boxes[:6]
        right_boxes = right_boxes[:6]

        left_crops = []
        right_crops = []

        for bbox in left_boxes:
            crop = image.crop(bbox)
            crop = self._apply_crop_transform(crop)
            left_crops.append(crop)

        for bbox in right_boxes:
            crop = image.crop(bbox)
            crop = self._apply_crop_transform(crop)
            right_crops.append(crop)

        # zero padding
        if len(left_crops) > 0:
            zero_crop = torch.zeros_like(left_crops[0])
        elif len(right_crops) > 0:
            zero_crop = torch.zeros_like(right_crops[0])
        else:
            zero_crop = torch.zeros(3, 224, 224)

        while len(left_crops) < 6:
            left_crops.append(zero_crop.clone())

        while len(right_crops) < 6:
            right_crops.append(zero_crop.clone())

        crops = left_crops + right_crops
        crops = torch.stack(crops, dim=0)
        # shape: [12, C, H, W]
        # first 6 = left side
        # last 6  = right side

        label = torch.tensor(item["target"], dtype=torch.long)

        return crops, label

    def _get_temporal_frame(self, item: Dict[str, Any]) -> Dict[str, Any]:
        frames = []
        for frame_id in item["frame_ids"]:
            path = self._img_path(item["video_id"], item["clip_id"], frame_id)
            frames.append(self._load_frame_image(path))

        frames = torch.stack(frames, dim=0)
        target = torch.tensor(item["target"], dtype=torch.long)
        return frames, target

    def _get_temporal_person(self, item: Dict[str, Any]) -> Dict[str, Any]:
        video_id = item["video_id"]
        clip_id = item["clip_id"]
        box_infos = item["box_infos"]
        crops = []
        label = None
        for box_info in box_infos:
            frame_id = box_info.frame_ID
            if frame_id == int(clip_id):
                label = box_info.category
            image_path = self._img_path(video_id, clip_id, frame_id)
            image = Image.open(image_path).convert("RGB")

            x1, y1, x2, y2 = box_info.box
            crop = image.crop((x1, y1, x2, y2))
            crop = self._apply_crop_transform(crop)
            crops.append(crop)
        frames = torch.stack(crops, dim=0)
        label = torch.tensor(PLAYER_LABELS[label], dtype=torch.long)
        return frames, label

    # Index building--------------------------------------------------------------------------
    def _build_frame_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples = []
        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            key_frame_id = int(clip_id)
            boxes = self._boxes_for_frame(clip_dict, key_frame_id)
            samples.append(self._base_item(video_id, clip_id, key_frame_id, clip_dict, boxes))
        return samples

    def _build_person_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples = []
        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            frame_id = int(clip_id)
            # the targeted frame and 1 frame after and before it
            for i in [-1, 0, 1]:
                path = self._img_path(video_id, clip_id, frame_id + i)
                boxes = self._boxes_for_frame(clip_dict, frame_id + i)
                for box_info in boxes:
                    if PLAYER_LABELS[box_info.category] == PLAYER_LABELS["standing"] and i != 0:
                        continue
                    samples.append({
                        "path": path,
                        "bbox": tuple(int(v) for v in box_info.box),
                        "target": PLAYER_LABELS[box_info.category],
                    })

        return samples

    def _build_frame_person_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples = []
        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            frame_id = int(clip_id)
            boxes = self._boxes_for_frame(clip_dict, frame_id)
            samples.append({
                "path": self._img_path(video_id, clip_id, frame_id),
                "boxes": [tuple(int(v) for v in b.box) for b in boxes],
                "target": GROUP_LABELS[clip_dict["category"]],
            })
        return samples

    def _build_temporal_person_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples = []
        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            player_boxes: Dict[int, List[Any]] = {}
            for boxes in clip_dict["frame_boxes_dct"].values():
                for box_info in self._filter_boxes(list(boxes)):
                    player_boxes.setdefault(int(box_info.player_ID), []).append(box_info)

            for player_id, box_infos in player_boxes.items():
                samples.append({
                    "video_id":  video_id,
                    "clip_id":   clip_id,
                    "player_id": player_id,
                    "box_infos": box_infos,
                })
        return samples

    def _build_temporal_clip_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        samples = []
        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            samples.append({
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_ids": self._sorted_frame_ids(clip_dict),
                "target": GROUP_LABELS[clip_dict["category"]],
                "clip_dict": clip_dict,
            })
        return samples

    # Helper Methods-----------------------------------------------------------

    def _iter_clips(self, video_ids: Sequence[str]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
        for video_id in video_ids:
            if video_id not in self.annotations:
                continue

            clip_ids = sorted(self.annotations[video_id].keys(), key=self._numeric_sort_key)
            for clip_id in clip_ids:
                yield video_id, clip_id, self.annotations[video_id][clip_id]

    def _img_path(self, video_id, clip_id, frame_id):
        path = os.path.join(self.videos_path, video_id, clip_id, str(frame_id) + ".jpg")
        return path

    def _base_item(self, video_id: str, clip_id: str, frame_id: int, clip_dict: Dict[str, Any],
                   boxes: Sequence[Any], ) -> Dict[str, Any]:
        group_label_name = clip_dict["category"]
        return {
            "mode": self.mode,
            "video_id": video_id,
            "clip_id": clip_id,
            "frame_id": frame_id,
            "path": self._img_path(video_id, clip_id, frame_id),
            "group_label": GROUP_LABELS[group_label_name],
            "group_label_name": group_label_name,
            "boxes": [self._box_to_dict(b) for b in boxes],
        }

    def _sorted_frame_ids(self, clip_dict: Dict[str, Any]) -> List[int]:
        return sorted(int(frame_id) for frame_id in clip_dict["frame_boxes_dct"].keys())

    def _boxes_for_frame(self, clip_dict: Dict[str, Any], frame_id: int) -> List[Any]:
        frame_boxes_dct = clip_dict["frame_boxes_dct"]
        boxes = list(frame_boxes_dct.get(int(frame_id), []))
        boxes = self._filter_boxes(boxes)
        return self._sort_boxes(boxes)

    def _filter_boxes(self, boxes: Sequence[Any]) -> List[Any]:
        filtered = []
        for box in boxes:
            if not self.include_lost and getattr(box, "lost", 0) != 0:
                continue
            if not self.include_generated and getattr(box, "generated", 0) != 0:
                continue
            filtered.append(box)
        return filtered

    def _sort_boxes(self, boxes: Sequence[Any]) -> List[Any]:
        if self.player_order == "player_id":
            return sorted(boxes, key=lambda b: (int(getattr(b, "player_ID", -1)), int(getattr(b, "frame_ID", -1))))

        # Spatial order: left-to-right, then top-to-bottom, then player_ID.
        return sorted(
            boxes,
            key=lambda b: (
                int(getattr(b, "box", (0, 0, 0, 0))[0]),
                int(getattr(b, "box", (0, 0, 0, 0))[1]),
                int(getattr(b, "player_ID", -1)),
            ),
        )

    def _box_to_dict(self, box_info: Any) -> Dict[str, Any]:
        label_name = box_info.category
        return {
            "player_id": int(box_info.player_ID),
            "frame_id": int(box_info.frame_ID),
            "bbox": tuple(int(v) for v in box_info.box),
            "label": PLAYER_LABELS[label_name],
            "label_name": label_name,
            "lost": int(getattr(box_info, "lost", 0)),
            "grouping": int(getattr(box_info, "grouping", 0)),
            "generated": int(getattr(box_info, "generated", 0)),
        }

    def _load_frame_image(self, path: str):
        image = Image.open(path).convert("RGB")
        return self._apply_frame_transform(image)

    def _apply_frame_transform(self, image):
        if self.frame_transform is not None:
            return self.frame_transform(image)
        return image

    def _apply_crop_transform(self, image):
        if self.crop_transform is not None:
            return self.crop_transform(image)
        return image

    @staticmethod
    def _numeric_sort_key(value: Any) -> Tuple[int, Any]:
        value_str = str(value)
        try:
            return 0, int(value_str)
        except ValueError:
            return 1, value_str
