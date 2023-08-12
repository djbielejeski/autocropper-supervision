import os
import cv2
import math

import supervision as sv

from scripts.utils.crop_utils import CropRatio


class AutoCropperImage:

    def __init__(
            self,
            person_results,
            face_results,
            crop_ratio=CropRatio(ratio=(3, 4)),
            person_percent_detection_cutoff=0.075,
            person_padding_percent=0.02
    ):
        self.file_name = os.path.basename(person_results.path)
        self.file_name_without_ext, self.file_extension = os.path.splitext(self.file_name)

        self.person_results = person_results
        self.face_results = face_results

        self.crop_ratio = crop_ratio
        self.person_percent_detection_cutoff = person_percent_detection_cutoff
        self.person_padding_percent = person_padding_percent

        self.height, self.width = self.person_results.orig_shape
        self.area = self.height * self.width
        self.min_person_area = self.area * self.person_percent_detection_cutoff

    def get_results(self):
        person_detections = self._get_person_bounding_boxes()
        face_detections = self._get_face_bounding_boxes()

        persons_with_faces_bounding_boxes = []

        # For each person check to see if we have a face detected
        for person in person_detections:
            for face in face_detections:
                if self._xyxy_contains(person, face):
                    centered_person = self._center(person, face)
                    persons_with_faces_bounding_boxes.append((centered_person, face))
                    break  # only do one face per person

        print(f"Found {len(persons_with_faces_bounding_boxes)} people with faces")

        return persons_with_faces_bounding_boxes

    def _center(self, person_xyxy, face_xyxy):
        person_left, person_top, person_right, person_bottom = person_xyxy
        person_height = person_bottom - person_top
        person_width = person_right - person_left

        # final output dimensions
        # find nearest division of 64
        output_width = output_height = 0
        if self.crop_ratio.height_value > self.crop_ratio.width_value:
            output_height = (person_height // 64) * 64
            output_width = self.crop_ratio.width_over_height * output_height

            if output_width > person_width:
                # If my output width is greater than our person width, we have to adjust our max sizes
                output_width = (person_width // 64) * 64
                output_height = output_width * self.crop_ratio.height_over_width
        else:
            output_width = (person_width // 64) * 64
            output_height = self.crop_ratio.height_over_width * output_width

            if output_height > person_height:
                output_height = (person_height // 64) * 64
                output_width = output_width * self.crop_ratio.width_over_height

        # find the center of the face
        face_left, face_top, face_right, face_bottom = face_xyxy
        face_height = face_bottom - face_top
        face_width = face_right - face_left

        face_center_x = face_left + math.floor(face_width / 2)
        face_center_y = face_top + math.floor(face_height / 2)

        potential_left = face_center_x - (output_width / 2)
        if potential_left < 0:
            potential_left = 0

        potential_right = potential_left + output_width

        potential_top = face_center_y - (output_height / 2)
        if potential_top < 0:
            potential_top = 0

        potential_bottom = potential_top + output_height

        final_left, final_right = self._get_bounding_box_centered_on_face(potential_left, potential_right, person_left,
                                                                          person_right, self.width)
        final_top, final_bottom = self._get_bounding_box_centered_on_face(potential_top, potential_bottom, person_top,
                                                                          person_bottom, self.height)

        final_bounding_box = (final_left, final_top, final_right, final_bottom)

        return final_bounding_box

    def _get_person_bounding_boxes(self):
        person_detections = sv.Detections.from_ultralytics(self.person_results)

        # Only grab people
        person_class_id = 0
        person_detections = person_detections[person_detections.class_id == person_class_id]
        person_detections = person_detections[person_detections.confidence > 0.5]
        person_detections = person_detections[person_detections.area > self.min_person_area]
        person_detections_xyxy = [self._standardize_xyxy(xyxy) for xyxy in person_detections.xyxy]
        # apply our padding
        person_detections_xyxy = [self._apply_padding(self.person_padding_percent, xyxy) for xyxy in
                                  person_detections_xyxy]

        return person_detections_xyxy

    def _get_face_bounding_boxes(self):
        face_detections = sv.Detections.from_ultralytics(self.face_results)
        face_detections = face_detections[face_detections.confidence > 0.5]
        face_detections_xyxy = [self._standardize_xyxy(xyxy) for xyxy in face_detections.xyxy]

        def sort_bounding_box(bounding_box):
            left, top, right, bottom = bounding_box
            area = (right - left) * (bottom - top)
            return area

        face_detections_xyxy.sort(key=sort_bounding_box, reverse=True)

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
        width = right - left
        height = bottom - top
        padding = math.ceil((width if width > height else height) * padding_percent)
        if padding > 0:
            left = left - padding if left - padding > 0 else 0
            top = top - padding if top - padding > 0 else 0
            right = right + padding if right + padding <= self.width else self.width
            bottom = bottom + padding if bottom + padding <= self.height else self.height
        return [left, top, right, bottom]

    def _within_bounds_start(self, start, person_start):
        return start >= 0 and start >= person_start

    def _within_bounds_end(self, end, person_end, maximum):
        return end <= maximum and end <= person_end

    def _within_bounds(self, start, end, person_start, person_end, maximum):
        return self._within_bounds_start(start, person_start) and self._within_bounds_end(end, person_end, maximum)

    def _get_bounding_box_centered_on_face(self, start, end, person_start, person_end, maximum):
        if start >= 0 and start >= person_start and end <= maximum and end <= person_end:
            pass
        else:
            # Walk 1 pixel
            shift = 1
            if end > maximum or end > person_end:
                # Walk 1 pixel backwards
                shift = -1

            iterations = 0
            while not self._within_bounds(start, end, person_start, person_end, maximum) and iterations < maximum:
                end += shift
                start += shift

                if start <= 0:
                    start = 0
                    iterations = maximum

                if end == maximum:
                    end = maximum
                    iterations = maximum

                iterations += 1

        return (start, end)


class ImageDirectoryLoader:
    def __init__(self, directory: str):
        self.image_paths = [
            os.path.join(directory, file_name) for file_name in os.listdir(directory)
            if any(file_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
        ]

        if len(self.image_paths) <= 0:
            raise Exception(f"No images (*.jpg, *.jpeg, *.png) found in '{directory}'.")
