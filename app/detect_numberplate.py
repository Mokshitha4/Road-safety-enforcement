
import subprocess
import uuid
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import pandas as pd

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