import torch
import os

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if ON_KAGGLE:
    pkl_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
    videos_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos"
    save_path = "/kaggle/working/"
else:
    pkl_path = "/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/annot_all.pkl"
    videos_path = "/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/videos"
    save_path = "/content/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
