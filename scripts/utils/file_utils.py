import os
import cv2


class ImageDirectoryLoader:

    def __init__(self, directory: str):
        self.image_names = [
            file_name for file_name in os.listdir(directory)
            if any(file_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
        ]

        if len(self.image_names) <= 0:
            raise Exception(f"No images (*.jpg, *.jpeg, *.png) found in '{directory}'.")

        self.image_paths = [
            os.path.join(directory, file_name) for file_name in self.image_names
        ]
        self.images = [cv2.imread(image_path) for image_path in self.image_paths]

        self.image_names_without_ext = []
        self.image_extensions = []

        for image_name in self.image_names:
            image_path_parts = os.path.splitext(image_name)
            self.image_names_without_ext.append(image_path_parts[0])
            self.image_extensions.append(image_path_parts[1])
