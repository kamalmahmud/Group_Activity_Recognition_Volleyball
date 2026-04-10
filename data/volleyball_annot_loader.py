import os
import pickle
from typing import List

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


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)
        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)
        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}
        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue
            # print(f'\t{clip_dir_path}')
            assert clip_dir in clip_category_dct

            file_annot = os.path.join(annot_root, video_dir, clip_dir, f'{clips_dir}.txt')
            frame_boxes_dict = load_tracking_annot(file_annot)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dict
            }

            videos_annot[video_dir] = clip_annot

        return videos_annot


def create_pkl_version():
    # You can use this function to create and save pkl version of the VolleyballDataset
    videos_root = f'{dataset_root}/videos'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation'

    videos_annot = load_volleyball_dataset(videos_root, annot_root)

    with open(f'{dataset_root}/annot_all.pkl', 'wb') as file:
        pickle.dump(videos_annot, file)


def test_pkl_version():
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)

    boxes: List[BoxInfo] = videos_annot['0']['13456']['frame_boxes_dct'][13454]
    print(boxes[0].category)
    print(boxes[0].box)


if __name__ == '__main__':
    annot_file = f'{dataset_root}/volleyball_tracking_annotation/4/24745/24745.txt'
    clip_dir_path = os.path.dirname(annot_file).replace('volleyball_tracking_annotation', 'videos')

    vis_clip(annot_file, clip_dir_path)
