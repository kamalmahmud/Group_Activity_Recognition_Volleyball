import torch

pkl_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
videos_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos"
save_path = "/kaggle/working/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")