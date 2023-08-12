import os
import math
import torch
from ultralytics import YOLO
import supervision as sv


class ModelLoader:

    def __init__(self, model_name="yolov8s.pt"):
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # https://docs.ultralytics.com/models/yolov8/#supported-modes
        model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        file_path = self.download_model(model_url, model_name)
        self.model = YOLO(file_path)
        self.model.to(device)

        # https://github.com/akanametov/yolov8-face
        face_model_name = "yolov8n-face.pt"
        face_model_url = f"https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/{face_model_name}"
        face_file_path = self.download_model(face_model_url, face_model_name)
        self.face_model = YOLO(face_file_path)
        self.face_model.to(device)

    def download_model(self, model_url: str, file_name: str):
        model_file_path = os.path.join(self.model_dir, file_name)

        if not os.path.exists(model_file_path):
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(model_url, model_dir=self.model_dir)

        return os.path.join(self.model_dir, file_name)

    def get_results(self, image_paths):
        person_results = self.model(image_paths)
        face_results = self.face_model(image_paths)

        merged_results = [(person_results[i], face_results[i]) for i in range(0, len(image_paths))]

        # Return a list of tuples, one tuple per file
        return merged_results
