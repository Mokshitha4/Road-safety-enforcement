from django.shortcuts import render
import subprocess
from django.shortcuts import render
import uuid
from PIL import Image
import os
import pandas as pd

# Render input_interface.html template
def render_inputs(request):
    return render(request,"input_interface.html")

# function to show image
def show_image(pathStr):
    img = Image.open(pathStr).convert("RGB")
    return img


# Generate a UUID
def generate_uuid():
    my_uuid = uuid.uuid4()
    return my_uuid


