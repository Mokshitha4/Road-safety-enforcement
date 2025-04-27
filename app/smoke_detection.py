# code snippet to detect vehicles emitting smoke using YOLOv9 model

from django.shortcuts import render
import subprocess
import uuid
from django.http import HttpResponse
import cv2
import numpy as np
import os 
import pandas as pd

# Generate a UUID
def generate_uuid():
    my_uuid = uuid.uuid4()
    return my_uuid

# Function to detect vehicles emitting smoke using YOLOv9 model
# This function takes an image path as input, generates a unique filename, and runs the YOLOv9 detection script with the specified parameters.
def detect_smoke(img_path):
    filename= generate_uuid()
    try:
        subprocess.run("python /yolov9/detect.py --img 1280 --conf 0.1 --device 0 --weights /models_yolov9/best_smoke.pt --source "+str(img_path)+" --save-crop --project /smoke_results --name "+str(filename), check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while detecting smoke: {e}")
        return None
    return os.path.join("/smoke_results/",str(filename))
detect_smoke("/inputs/Auto_pol.jpeg")


