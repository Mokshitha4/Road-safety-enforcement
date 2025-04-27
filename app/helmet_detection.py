# code to detect helmets using YOLOv9 model

from django.shortcuts import render
import subprocess
import uuid
import cv2
import os
import pandas as pd

# Generate a UUID
def generate_uuid():
    my_uuid = uuid.uuid4()
    return my_uuid

#Function to detect helmets using YOLOv9 model
# This function takes an image path as input, generates a unique filename, and runs the YOLOv9 detection script with the specified parameters.
# The detected results are saved in a specified directory, and the function returns the path to the saved results.
def detect_helmet(img_path):
    filename= generate_uuid()
    try:
        subprocess.run("python yolov9/detect.py --img 1280 --conf 0.1 --device 0 --weights /models_yolov9/best_helmet.pt --source "+str(img_path)+" --save-crop --project /helmet_results --name "+str(filename), check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while detecting helmet: {e}")
        return None
    return os.path.join("/helmet_results/",str(filename))