import os
from ultralytics import YOLO


class ModelLoader:

    def __init__(self, model_name="yolov8s.pt"):
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # https://docs.ultralytics.com/models/yolov8/#supported-modes
        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        file_path = self.download_model(model_url, model_name)
        self.model = YOLO(file_path)

        # https://github.com/akanametov/yolov8-face
        face_model_name = "yolov8n-face.pt"
        face_model_url = f"https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/{face_model_name}"
        face_file_path = self.download_model(face_model_url, face_model_name)
        self.face_model = YOLO(face_file_path)

    def download_model(self, model_url: str, file_name: str):
        model_file_path = os.path.join(self.model_dir, file_name)

        if not os.path.exists(model_file_path):
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(model_url, model_dir=self.model_dir)

        return os.path.join(self.model_dir, file_name)
