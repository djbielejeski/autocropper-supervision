import os
import cv2
from ultralytics import YOLO
import supervision as sv


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# https://docs.ultralytics.com/models/yolov8/#supported-modes
model_name = "yolov8s.pt"
model_url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
model_file_path = os.path.join(model_dir, model_name)

if not os.path.exists(model_file_path):
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url(model_url, model_dir=model_dir)


image = cv2.imread("image.jpg")
model = YOLO(model_file_path)
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)

# Only grab people
detections = detections[detections.class_id == 0]
detections = detections[detections.confidence > 0.5]

print(detections)

classes = ['person']

with sv.ImageSink(target_dir_path=results_folder) as sink:
    for i, xyxy in enumerate(detections.xyxy):
        cropped_image = sv.crop(image=image.copy(), xyxy=xyxy)
        sink.save_image(image=cropped_image, image_name=f"image_{i:05d}.png")



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