import os
from ultralytics import YOLO


class ModelLoader:

    def __init__(self, model_name="yolov8s.pt"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        model_file_path = os.path.join(model_dir, model_name)

        # https://docs.ultralytics.com/models/yolov8/#supported-modes
        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"

        if not os.path.exists(model_file_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(model_url, model_dir=model_dir)

        file_path = os.path.join(model_dir, model_name)
        self.model = YOLO(file_path)
