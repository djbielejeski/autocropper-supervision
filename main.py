import os
import cv2

import supervision as sv

from scripts.utils.model_utils import ModelLoader
from scripts.utils.file_utils import ImageDirectoryLoader

results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# Load our model
model_loader = ModelLoader(model_name="yolov8s.pt")

# Parse our images
images_directory = "C:/Images"
image_directory_loader = ImageDirectoryLoader(directory=images_directory)

print(f"Processing {len(image_directory_loader.image_paths)} images from '{images_directory}'")

for i, ac_image in enumerate(image_directory_loader.images):
    print(f"Processing {ac_image.file_path}...")

    # run detection
    persons_with_faces = ac_image.get_results(model_loader)

    for person_index, person_with_face in enumerate(persons_with_faces):
        person_xyxy, face_xyxy = person_with_face

        # Center the image around the face detection for this item - TODO
        person_image = sv.crop(image=ac_image.image.copy(), xyxy=person_xyxy)

        for dimension in ac_image.crop_ratio.dimensions:
            width, height = dimension
            width_height_text = f"{width}x{height}"

            with sv.ImageSink(target_dir_path=os.path.join(results_folder, width_height_text)) as sink:
                image_name = f"{ac_image.image_name_without_ext}_{person_index:02d}_{width_height_text}{ac_image.image_extension}"
                resized_image = cv2.resize(person_image, (width, height))
                sink.save_image(image=resized_image, image_name=image_name)

                # face_image = sv.crop(image=ac_image.image.copy(), xyxy=face_xyxy)
                # sink.save_image(image=face_image, image_name=f"{ac_image.image_name_without_ext}_face_{person_index:02d}{ac_image.image_extension}")

print("Complete")

# box_annotator = sv.BoxAnnotator()
# labels = [
#    f"{classes[class_id]} {confidence:0.2f}"
#    for _, _, confidence, class_id, _
#    in detections
# ]
# annotated_image = box_annotator.annotate(
#     scene=image.copy(),
#     detections=detections,
#     labels=labels
# )
#
# output_file = os.path.join(results_folder, f"annotated.jpg")
# cv2.imwrite(output_file, annotated_image)
