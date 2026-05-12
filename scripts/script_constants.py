import torch
import os

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if ON_KAGGLE:
    pkl_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
    videos_path = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/videos"
    save_path = "/kaggle/working/"
    player_temporal_checkpoint_path = "/kaggle/input/models/kamalalqedra/temporal-player-action/pytorch/default/1/best_model.pth"
else:
    pkl_path = "/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/annot_all.pkl"
    videos_path = "/root/.cache/kagglehub/datasets/sherif31/group-activity-recognition-volleyball/versions/1/videos"
    save_path = "/content/"
    player_temporal_checkpoint_path = "/content/best_model_player.pth"
    checkpoint_b7 = "/content/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
