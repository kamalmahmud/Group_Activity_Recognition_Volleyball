from typing import Any, Dict

import torch
from PIL import Image

from .constants import GROUP_LABELS, PLAYER_LABELS


class DatasetGettersMixin:
    def _get_frame(self, item: Dict[str, Any]):
        image = self._load_frame_image(item["path"])
        label = GROUP_LABELS[item["group_label_name"]]
        return image, label

    def _get_person(self, item: Dict[str, Any]):
        image = Image.open(item["path"]).convert("RGB")

        crop = image.crop(item["bbox"])
        crop = self._apply_crop_transform(crop)

        label = item["target"]

        return crop, label

    def _get_frame_person(self, item: Dict[str, Any]):
        image = Image.open(item["path"]).convert("RGB")

        width, _ = image.size
        boxes = item["boxes"]

        left_boxes = []
        right_boxes = []

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2

            if center_x < width / 2:
                left_boxes.append(bbox)
            else:
                right_boxes.append(bbox)

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

        label = torch.tensor(item["target"], dtype=torch.long)

        return crops, label

    def _get_temporal_frame(self, item: Dict[str, Any]):
        frames = []

        for frame_id in item["frame_ids"]:
            path = self._img_path(
                item["video_id"],
                item["clip_id"],
                frame_id,
            )
            frames.append(self._load_frame_image(path))

        frames = torch.stack(frames, dim=0)
        target = torch.tensor(item["target"], dtype=torch.long)

        return frames, target

    def _get_temporal_person(self, item: Dict[str, Any]):
        video_id = item["video_id"]
        clip_id = item["clip_id"]
        box_infos = item["box_infos"]

        crops = []
        label_name = None

        for box_info in box_infos:
            frame_id = box_info.frame_ID

            if frame_id == int(clip_id):
                label_name = box_info.category

            image_path = self._img_path(video_id, clip_id, frame_id)
            image = Image.open(image_path).convert("RGB")

            x1, y1, x2, y2 = box_info.box
            crop = image.crop((x1, y1, x2, y2))
            crop = self._apply_crop_transform(crop)

            crops.append(crop)

        frames = torch.stack(crops, dim=0)

        if label_name is None:
            raise ValueError(
                f"Could not find target label for video={video_id}, clip={clip_id}"
            )

        label = torch.tensor(PLAYER_LABELS[label_name], dtype=torch.long)

        return frames, label

    def _get_temporal_person_clip(self, item):
        list_of_frames, label = item
        max_players = 12
        frames = []
        for frame in list_of_frames:
            image = Image.open(frame["frame_path"]).convert("RGB")
            crops = []
            for box in frame["boxes"]:
                crop = image.crop(box)
                crop = self._apply_crop_transform(crop)
                crops.append(crop)

            crops = torch.stack(crops, dim=0)
            num_players = crops.shape[0]
            # pad with last image if less than 12
            if num_players < 12:
                pad = crops[-1:].repeat(max_players - num_players, 1, 1, 1)
                crops = torch.cat([crops, pad], dim=0)

            frames.append(crops)

        frames = torch.stack(frames, dim=0)

        # change to [12, T, C, H, W]
        frames = frames.permute(1, 0, 2, 3, 4)

        target = torch.tensor(label, dtype=torch.long)
        return frames, target
