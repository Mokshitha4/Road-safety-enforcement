
#Sample code to detect vehicles using YOLOv8 model

from ultralytics import YOLO

model = YOLO('/model_yolov8/FGVD_vehicle_detcetion.pt')  # pretrained YOLOv8n model


# Run batched inference on a list of images
results = model(['/inputs/Auto_pol.jpeg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
result.save(filename='/results/detection_img.jpg') 