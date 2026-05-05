from typing import Any, Dict, List, Sequence

from .constants import GROUP_LABELS, PLAYER_LABELS


class DatasetIndexBuildersMixin:
    def _build_frame_index(
            self,
            video_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            key_frame_id = int(clip_id)
            boxes = self._boxes_for_frame(clip_dict, key_frame_id)

            samples.append(
                self._base_item(
                    video_id=video_id,
                    clip_id=clip_id,
                    frame_id=key_frame_id,
                    clip_dict=clip_dict,
                    boxes=boxes,
                )
            )

        return samples

    def _build_person_index(
            self,
            video_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            frame_id = int(clip_id)

            for offset in [-1, 0, 1]:
                current_frame_id = frame_id + offset
                path = self._img_path(video_id, clip_id, current_frame_id)
                boxes = self._boxes_for_frame(clip_dict, current_frame_id)

                for box_info in boxes:
                    is_standing = (
                            PLAYER_LABELS[box_info.category]
                            == PLAYER_LABELS["standing"]
                    )

                    if is_standing and offset != 0:
                        continue

                    samples.append(
                        {
                            "path": path,
                            "bbox": tuple(int(v) for v in box_info.box),
                            "target": PLAYER_LABELS[box_info.category],
                        }
                    )

        return samples

    def _build_frame_person_index(
            self,
            video_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            frame_id = int(clip_id)
            boxes = self._boxes_for_frame(clip_dict, frame_id)

            samples.append(
                {
                    "path": self._img_path(video_id, clip_id, frame_id),
                    "boxes": [tuple(int(v) for v in box.box) for box in boxes],
                    "target": GROUP_LABELS[clip_dict["category"]],
                }
            )

        return samples

    def _build_temporal_person_clip_index(self, video_ids):
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            frames = []

            for frame_id in sorted(clip_dict["frame_boxes_dct"]):
                boxes = self._boxes_for_frame(clip_dict, frame_id)

                frames.append({
                    "frame_id": frame_id,
                    "boxes": boxes,
                    "frame_path": self._img_path(video_id, clip_id, frame_id),
                })

            label = GROUP_LABELS[clip_dict["category"]]
            samples.append((frames, label))

        return samples

    def _build_temporal_clip_index(
            self,
            video_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            samples.append(
                {
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "frame_ids": self._sorted_frame_ids(clip_dict),
                    "target": GROUP_LABELS[clip_dict["category"]],
                    "clip_dict": clip_dict,
                }
            )

        return samples

    def _build_temporal_person_index(
            self,
            video_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        for video_id, clip_id, clip_dict in self._iter_clips(video_ids):
            player_boxes: Dict[int, List[Any]] = {}

            for boxes in clip_dict["frame_boxes_dct"].values():
                for box_info in self._filter_boxes(list(boxes)):
                    player_id = int(box_info.player_ID)
                    player_boxes.setdefault(player_id, []).append(box_info)

            for player_id, box_infos in player_boxes.items():
                box_infos = self._sort_boxes(box_infos)

                samples.append(
                    {
                        "video_id": video_id,
                        "clip_id": clip_id,
                        "player_id": player_id,
                        "box_infos": box_infos,
                    }
                )

        return samples
