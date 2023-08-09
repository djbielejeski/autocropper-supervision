import os
import cv2
import math

import supervision as sv

from scripts.utils.model_utils import ModelLoader


class AutoCropperImage:
    person_detections = []
    face_detections = []

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.image = cv2.imread(file_path)
        self.height, self.width, _ = self.image.shape
        self.image_name_without_ext, self.image_extension = os.path.splitext(os.path.basename(file_path))
        self.person_padding_percent = 0.01

    def get_results(self, model_loader: ModelLoader):
        self.person_detections = self._get_person_bounding_boxes(model_loader)
        self.face_detections = self._get_face_bounding_boxes(model_loader)

        persons_with_faces_bounding_boxes = []

        # For each person check to see if we have a face detected
        for i, person in enumerate(self.person_detections):
            for j, face in enumerate(self.face_detections):
                if self._xyxy_contains(person, face):
                    persons_with_faces_bounding_boxes.append((person, face))

        print(f"Found {len(persons_with_faces_bounding_boxes)} people with faces")

        # iterate over all of the persons with faces and do the crops
        # TODO

        return persons_with_faces_bounding_boxes

    def _get_person_bounding_boxes(self, model_loader: ModelLoader):
        result = model_loader.model(self.image)[0]
        person_detections = sv.Detections.from_ultralytics(result)

        # Only grab people
        person_class_id = 0
        person_detections = person_detections[person_detections.class_id == person_class_id]
        person_detections = person_detections[person_detections.confidence > 0.5]
        person_detections_xyxy = [self._standardize_xyxy(xyxy) for xyxy in person_detections.xyxy]
        # apply our padding
        person_detections_xyxy = [self._apply_padding(self.person_padding_percent, xyxy) for xyxy in person_detections_xyxy]

        return person_detections_xyxy

    def _get_face_bounding_boxes(self, model_loader: ModelLoader):
        result_face = model_loader.face_model(self.image)[0]
        face_detections = sv.Detections.from_ultralytics(result_face)
        face_detections = face_detections[face_detections.confidence > 0.5]
        face_detections_xyxy = [self._standardize_xyxy(xyxy) for xyxy in face_detections.xyxy]

        return face_detections_xyxy

    def _standardize_xyxy(self, xyxy):
        left, top, right, bottom = xyxy  # [100, 100, 600, 700]
        left = math.floor(left)
        top = math.floor(top)
        right = math.ceil(right)
        bottom = math.ceil(bottom)
        return [left, top, right, bottom]

    def _xyxy_contains(self, xyxy1, xyxy2):
        left_1, top_1, right_1, bottom_1 = xyxy1
        left_2, top_2, right_2, bottom_2 = xyxy2

        return left_1 <= left_2 and top_1 <= top_2 and right_1 >= right_2 and bottom_1 >= bottom_2

    def _apply_padding(self, padding_percent, xyxy):
        left, top, right, bottom = xyxy  # [100, 100, 600, 700]
        padding = math.ceil((self.width if self.width > self.height else self.height) * padding_percent)
        print(f"padding: {padding}")
        if padding > 0:
            left -= padding if left > padding else 0
            top -= padding if top > padding else 0
            right += padding if right + padding <= self.width else self.width
            bottom += padding if bottom + padding <= self.height else self.height
        return [left, top, right, bottom]


class ImageDirectoryLoader:
    def __init__(self, directory: str):
        self.image_paths = [
            os.path.join(directory, file_name) for file_name in os.listdir(directory)
            if any(file_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
        ]

        if len(self.image_paths) <= 0:
            raise Exception(f"No images (*.jpg, *.jpeg, *.png) found in '{directory}'.")

        self.images = [AutoCropperImage(file_path=image_path) for image_path in self.image_paths]
