
from django.shortcuts import render
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

# Function to perform OCR on the image using TrOCR
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

# Function to detect vehicles numberplates using YOLOv9 model
# This function takes an image path as input, generates a unique filename, and runs the YOLOv9 detection script with the specified parameters.  
def detect_numberplate(img_path):
    filename = generate_uuid()
    try:
        subprocess.run("python yolov9/detect.py --img 1280 --conf 0.1 --device 0 --weights models_yolov9/best_numberplate.pt --source " + str(img_path) + " --save-crop --project " + str("results") + " --name " + str(filename), check=True)  
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while detecting number plate: {e}")
        return None
    return os.path.join("results/numberPlates_results/", str(filename))


def perform_OCR(img_path):
    image = show_image(img_path)
    image1 = image.crop((0, 10, image.size[0], 40))
    strOCR=str(ocr_image(image1))
    print(strOCR)
    return strOCR 
