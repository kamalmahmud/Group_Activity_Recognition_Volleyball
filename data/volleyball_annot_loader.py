import os

import cv2

from boxinfo import BoxInfo

dataset_root = ''


def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)

            if box_info.player_ID > 11:
                continue

            player_boxes[box_info.player_ID].append(box_info)
            # let's create view from frame to boxes
            for player_id, boxes_info in player_boxes.items():
                boxes_info = boxes_info[5:-6]

                for box_info in boxes_info:

                    if box_info.frame_ID not in frame_boxes_dct:
                        frame_boxes_dct[box_info.frame_ID] = []

                    frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct


def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f'{frame_id}.jpg')
        img = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(img, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(180)
    cv2.destroyAllWindows()
