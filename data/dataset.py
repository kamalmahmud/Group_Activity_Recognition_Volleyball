import os
import os.path
import pickle
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


class VolleyballDataset(Dataset):
    def __init__(
            self,
            pkl_path: str,
            videos_path: str,
            transforms=None,
            frame_transform=None,
            crop_transform=None,
            split: str = "train",
            mode: str = "frame",
            player_order: str = "player_id",
            include_lost: bool = True,
            include_generated: bool = True,
    ):
        self.videos_path = videos_path
        self.frame_transform = frame_transform if frame_transform is not None else transforms
        self.crop_transform = crop_transform if crop_transform is not None else transforms
        self.pkl_path = pkl_path
        self.split = split
        self.mode = mode
        self.player_order = player_order
        self.include_lost = include_lost
        self.include_generated = include_generated
        self.splits = {
            'train': ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                      "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54", ],
            'val': ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"],
            'test': ['4', '5', '9', '11', '14', '20', '21', '25', '29', '34', '35', '37', '43', '44', '45', '47']
        }
        self.labels = {
            'l-pass': 0, 'r-pass': 1, 'l-spike': 2, 'r_spike': 3,
            'l_set': 4, 'r_set': 5, 'l_winpoint': 6, 'r_winpoint': 7
        }
        self.player_labels = {
            'blocking': 0, 'digging': 1, 'falling': 2, 'jumping': 3,
            'moving': 4, 'setting': 5, 'spiking': 6, 'standing': 7, 'waiting': 8
        }
        with open(self.pkl_path, "rb") as f:
            self.annotations = pickle.load(f)

        self.samples = self._build_index(self.splits[split])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.mode == "frame":
            return self._get_frame(item)
        if self.mode == "person":
            return self._get_person(item)
        raise RuntimeError(f"Unhandled mode={self.mode!r}")

    # __getitem__ Methods-------------------------------------------------------------------------------
    def _get_frame(self, item):
        image = self._load_frame_image(item["path"])
        label = self.labels[item["group_label_name"]]
        return image, label

    def _get_person(self, item: Dict[str, Any]) -> Dict[str, Any]:
        image = Image.open(item["path"]).convert("RGB")
        crop = image.crop(item["bbox"])
        crop = self._apply_crop_transform(crop)
        label = item["target"]
        return crop, label

        # Index building Methods------------------------------------------------------

    def _build_index(self, video_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if self.mode == "frame":
            return self._build_frame_index(video_ids)
        elif self.mode == "person":
            return self._build_person_index(video_ids)
        else:
            raise ValueError(f"Unknown mode={self.mode!r}.")

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
            # the targeted frame and 1 frames after it
            for i in [-1, 0, 1]:
                path = self._img_path(video_id, clip_id, frame_id + i)
                boxes = self._boxes_for_frame(clip_dict, frame_id + i)
                for box_info in boxes:
                    if self.player_labels[box_info.category] in 7 and i != 0:
                        continue
                    samples.append({
                        "path": path,
                        "bbox": tuple(int(v) for v in box_info.box),
                        "target": self.player_labels[box_info.category],
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
            "group_label": self.labels[group_label_name],
            "group_label_name": group_label_name,
            "boxes": [self._box_to_dict(b) for b in boxes],
        }

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
            "label": self.player_labels[label_name],
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
