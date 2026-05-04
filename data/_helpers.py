import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from .constants import GROUP_LABELS, PLAYER_LABELS


class DatasetHelpersMixin:
    def _iter_clips(
            self,
            video_ids: Sequence[str],
    ) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
        for video_id in video_ids:
            if video_id not in self.annotations:
                continue

            clip_ids = sorted(
                self.annotations[video_id].keys(),
                key=self._numeric_sort_key,
            )

            for clip_id in clip_ids:
                yield video_id, clip_id, self.annotations[video_id][clip_id]

    def _img_path(self, video_id: str, clip_id: str, frame_id: int) -> str:
        return os.path.join(
            self.videos_path,
            video_id,
            clip_id,
            f"{frame_id}.jpg",
        )

    def _base_item(
            self,
            video_id: str,
            clip_id: str,
            frame_id: int,
            clip_dict: Dict[str, Any],
            boxes: Sequence[Any],
    ) -> Dict[str, Any]:
        group_label_name = clip_dict["category"]

        return {
            "mode": self.mode,
            "video_id": video_id,
            "clip_id": clip_id,
            "frame_id": frame_id,
            "path": self._img_path(video_id, clip_id, frame_id),
            "group_label": GROUP_LABELS[group_label_name],
            "group_label_name": group_label_name,
            "boxes": [self._box_to_dict(box) for box in boxes],
        }

    def _sorted_frame_ids(self, clip_dict: Dict[str, Any]) -> List[int]:
        return sorted(
            int(frame_id)
            for frame_id in clip_dict["frame_boxes_dct"].keys()
        )

    def _boxes_for_frame(
            self,
            clip_dict: Dict[str, Any],
            frame_id: int,
    ) -> List[Any]:
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
            return sorted(
                boxes,
                key=lambda box: (
                    int(getattr(box, "player_ID", -1)),
                    int(getattr(box, "frame_ID", -1)),
                ),
            )

        return sorted(
            boxes,
            key=lambda box: (
                int(getattr(box, "box", (0, 0, 0, 0))[0]),
                int(getattr(box, "box", (0, 0, 0, 0))[1]),
                int(getattr(box, "player_ID", -1)),
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
