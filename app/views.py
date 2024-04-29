from django.shortcuts import render
# from yolov9.detect import run
# Create your views here.
import subprocess
from django.shortcuts import render
import uuid
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import pandas as pd

#load transformer OCR model
def render_inputs(request):
    return render(request,"input_interface.html")

def show_image(pathStr):
    img = Image.open(pathStr).convert("RGB")
    return img

def ocr_image(src_img):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed") 
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True,max_length=20)[0]

# Generate a UUID
def generate_uuid():
    my_uuid = uuid.uuid4()
    return my_uuid

def detect_numberplate(img_path):
    filename= generate_uuid()
    # run(source=img_path,weights='C:/Users/mmandadi/work_proj/django_project/final_yr_project/models_yolov9/best_numberplate.pt',save_crop=True,name='exp',project='C:/Users/mmandadi/work_proj/django_project/final_yr_project/results')
    subprocess.run("python C:/Users/mmandadi/work_proj/django_project/final_yr_project/django_app/yolov9/detect.py --img 1280 --conf 0.1 --device 0 --weights C:/Users/mmandadi/work_proj/django_project/final_yr_project/models_yolov9/best_numberplate.pt --source "+str(img_path)+" --save-crop --project C:/Users/mmandadi/work_proj/django_project/final_yr_project/results --name "+str(filename))
    return os.path.join("C:/Users/mmandadi/work_proj/django_project/final_yr_project/numberPlates_results/",str(filename))
    

def perform_OCR(img_path):
    image = show_image(img_path)
    image1 = image.crop((0, 10, image.size[0], 40))
    strOCR=str(ocr_image(image1))
    print(strOCR)
    return strOCR 
# detect_numberplate('C:/Users/mmandadi/work_proj/django_project/final_yr_project/results/detections_cropped_expanded_0_car.jpg')
# perform_OCR("C:/Users/mmandadi/work_proj/django_project/final_yr_project/results/d9dafa04-a8fa-4e6c-92aa-cd9d04eb3559/crops/License-plate/car_img4.jpg")
# detect_numberplate('car_img.jpg')




# def save_cropped_image(image_path, result, save_dir="vehicle_cropped_images"):
#     # # Process results list
#     # for result in results:
#     #     print(result)
#     #     boxes = result.boxes  # Boxes object for bounding box outputs
#     #     masks = result.masks  # Masks object for segmentation masks outputs
#     #     keypoints = result.keypoints  # Keypoints object for pose outputs
#     #     probs = result.probs  # Probs object for classification outputs
#     #     print(boxes,masks,keypoints, probs)
#     img = cv2.imread(image_path)  # Load the original image
#     if img is None:
#         print(f"Error: Could not read image {image_path}")
#         return

#     # Create the save directory if it doesn't exist
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # Access bounding box information directly from the Results object
#     boxes= result[0].boxes
#     print(boxes)
#     for idx, (x1, y1, x2, y2, conf, class_id, name) in enumerate(boxes.xyxy[0]):
#         # Extract bounding box coordinates and ensure they are within image bounds
#         x1 = max(0, int(x1.item()))  # Clamp x1 to avoid going out of image
#         y1 = max(0, int(y1.item()))  # Clamp y1 to avoid going out of image
#         x2 = min(img.shape[1], int(x2.item()))  # Clamp x2 to image width
#         y2 = min(img.shape[0], int(y2.item()))  # Clamp y2 to image height

#         cropped_img = img[y1:y2, x1:x2]  # Extract the cropped image

#         # Construct the filename with class name and detection index
#         filename = f"{save_dir}/{name}_{idx}.jpg"

#         # Save the cropped image
#         cv2.imwrite(filename, cropped_img)
#         print(f"Saved cropped image: {filename}")



# def detect_vehicle(img_path):
#     results = model.predict(source=img_path,save_crop=True)  # Run YOLOv8 prediction on the image
#     saved_dir=results[0].save_dir
#     motorcycle_dir=os.path.join(saved_dir,"crops/motorcycle")
#     motorcycle_status=os.path.exists(motorcycle_dir) 
#     scooter_dir=os.path.join(saved_dir,"crops/scooter")
#     scooter_status=os.path.exists(scooter_dir) 
#     #check for two wheelers: motorcycle,scooter
#     if motorcycle_status:
#         for filename in os.listdir(motorcycle_dir):
#             print(filename)
#             #helmet detection module
#             #smoke detecion module
#     if scooter_status:
#         for filename in os.listdir(scooter_dir):
#             print(filename)
#             #helmet detection module
#             #smoke detecion module
#     car_dir=os.path.join(saved_dir,"crops/car")
#     if os.path.exists(car_dir) :
#         for filename in os.listdir(car_dir):
#             print(filename)
#             #smoke detecion module
#     truck_dir=os.path.join(saved_dir,"crops/truck")
#     if os.path.exists(truck_dir) :
#         for filename in os.listdir(truck_dir):
#             print(filename)
#             #smoke detecion module
#     autorickshaw_dir=os.path.join(saved_dir,"crops/autorickshaw")
#     if os.path.exists(autorickshaw_dir) :
#         for filename in os.listdir(autorickshaw_dir):
#             print(filename)
#             #smoke detecion module
#     mini_bus_dir=os.path.join(saved_dir,"crops/mini-bus")
#     if os.path.exists(mini_bus_dir) :
#         for filename in os.listdir(mini_bus_dir):
#             print(filename)
#             #smoke detecion module
#     bus_dir=os.path.join(saved_dir,"crops/bus")
#     if os.path.exists(bus_dir) :
#         for filename in os.listdir(bus_dir):
#             print(filename)
#             #smoke detecion module
#     #0: 'car', 1: 'motorcycle', 2: 'scooter', 3: 'truck', 4: 'autorickshaw', 5: 'mini-bus', 6: 'bus'
        
#     # save_cropped_image(img_path, results)

# detect_vehicle('C:/Users/mmandadi/work_proj/django_project/final_yr_project/inputs/img50.jpeg')
            
