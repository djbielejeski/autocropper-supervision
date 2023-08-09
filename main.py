import os
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

print(f"Processing {len(image_directory_loader.image_names)} images from '{images_directory}'")

for i, image in enumerate(image_directory_loader.images):
    result = model_loader.model(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Only grab people
    person_class_id = 0
    detections = detections[detections.class_id == person_class_id]
    detections = detections[detections.confidence > 0.5]

    original_file_name = image_directory_loader.image_names_without_ext[i]
    original_file_ext = image_directory_loader.image_extensions[i]

    print(f"Processing {original_file_name}{original_file_ext}...")

    with sv.ImageSink(target_dir_path=results_folder) as sink:
        for person_index, xyxy in enumerate(detections.xyxy):
            cropped_image = sv.crop(image=image.copy(), xyxy=xyxy)
            sink.save_image(image=cropped_image, image_name=f"{original_file_name}_{person_index:02d}{original_file_ext}")


print("Complete")

#box_annotator = sv.BoxAnnotator()
#labels = [
#    f"{classes[class_id]} {confidence:0.2f}"
#    for _, _, confidence, class_id, _
#    in detections
#]
#annotated_image = box_annotator.annotate(
#     scene=image.copy(),
#     detections=detections,
#     labels=labels
#)
#
#output_file = os.path.join(results_folder, f"annotated.jpg")
#cv2.imwrite(output_file, annotated_image)