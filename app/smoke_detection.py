from django.shortcuts import render
# from yolov9.detect import run
# Create your views here.
import subprocess
import uuid
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from ultralytics import YOLO
from django.http import HttpResponse
import cv2
import numpy as np
import os 
import pandas as pd
# Generate a UUID
def generate_uuid():
    my_uuid = uuid.uuid4()
    return my_uuid

def detect_smoke(img_path):
    filename= generate_uuid()
    subprocess.run("python C:/Users/mmandadi/work_proj/django_project/final_yr_project/yolov9/detect.py --img 1280 --conf 0.1 --device 0 --weights C:/Users/mmandadi/work_proj/django_project/final_yr_project/models_yolov9/best_smoke.pt --source "+str(img_path)+" --save-crop --project C:/Users/mmandadi/work_proj/django_project/final_yr_project/smoke_results --name "+str(filename))
    return os.path.join("C:/Users/mmandadi/work_proj/django_project/final_yr_project/smoke_results/",str(filename))
detect_smoke("C:/Users/mmandadi/work_proj/django_project/final_yr_project/inputs/Auto_pol.jpeg")


