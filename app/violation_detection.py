import cv2
import numpy as np
from ultralytics import YOLO
from django.shortcuts import render
import os
import json
from django_app.helmet_detection import detect_numberplate, perform_OCR
from django.http import HttpResponse
import cv2
import torch
from django_app.helmet_detection import detect_helmet
from django_app.smoke_detection import detect_smoke

model = YOLO('model_yolov8/FGVD_vehicle_detcetion.pt')  # trained YOLOv8 model

def expand_bbox_to_top(bbox, image_shape, expansion_factor=0.1):
    """
    Expands a bounding box to the top of the image by a specified factor.

    Args:
        bbox (dict): YOLOv8 prediction dictionary containing bounding box information.
        image_shape (tuple): Image shape (height, width).
        expansion_factor (float, optional): Factor for expanding the bounding box. Defaults to 0.2.

    Returns:
        tuple or None: Modified bounding box coordinates (x_min, y_min, x_max, y_max) or None if invalid data.
    """
    # Move bbox dictionary and image to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bbox = bbox.to(device)

    # Resize image to expected format (adjust based on your model's requirements)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))  # Assuming model expects 640x640
    image = torch.from_numpy(image).to(device)

    xyxy= bbox.xyxy # Access xywh coordinates
    x1,y1,x2,y2 = xyxy[0] # Move xywhs to CPU for calculations

    height, width = image_shape[:2]

    # Calculate expansion distance based on image height and factor, ensuring non-negative value
    expansion_distance = max(0, int(height * expansion_factor))

    # Ensure new y_min stays within image bounds (clamp to 0)
    new_y1 = max(0, y1- expansion_distance)

    return (int(x1), int(new_y1), int(x2), int(y2),height)


def crop_image_detect_violations(image, model, save_dir, expansion_factor):
    """
    Crops and expands an image based on YOLOv8 predictions using GPU if available. Then detects violations for each detected vehicle.

    Args:
        image_path (str): Path to the image.
        model: Loaded YOLOv8 model.
        save_dir (str): Directory to save the cropped and expanded images.
        expansion_factor (float, optional): Factor for expanding the bounding box. Defaults to 0.2.
    output: A dictionary containing counts of detected vehicles and violations.
    """

    image_path="detections"
    # Move image to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.from_numpy(image).to(device)
    image=image.cpu().numpy()
    print(image.shape)
    # Get predictions from YOLOv8 (adjust based on your model's output format)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):  # Enable automatic mixed precision for potentially faster inference on compatible GPUs
        results = model.predict(image)  # Move image back to CPU for prediction (might need adjustment)
    boxes = results[0].boxes  # Assuming boxes are in results[0]
    print(boxes)
    names= results[0].names
    images=[]
    two_wheelers=[]
    other_vehicles=[]
    classes=boxes.cls
    two_wheelers_count=0
    other_vehicles_count=0
    smoke_count= 0
    without_helmet_count=0
    with_helmet_count =0
    for i, box in enumerate(boxes):
        # Check for valid dictionary format and presence of 'xywh' ke
        # Expand bounding box (handle potential errors)

        expanded_box = expand_bbox_to_top(box.to(device), image.shape, expansion_factor)  # Move box to GPU for calculations
        if expanded_box is None:
            continue  # Skip to next bounding box if error occurs

        # Extract ROI (region of interest) based on expanded box
        x1,y1,x2,y2,height = expanded_box
        x1 = max(0, x1)  # Clamp x1 to 0 (avoid negative values)
        y1 = max(0, y1)  # Clamp y1 to 0 (avoid negative values) 
        y2 = min(height, y2)  # Clamp y2 to image height (avoid exceeding image boundaries)

        # Crop the image using the extracted coordinates

        cropped_image = image[y1:y2,x1:x2] # Move cropped image back to CPU

        # Generate filename based on original filename and index (optional)
        filename, _ = os.path.splitext(os.path.basename(image_path))
        cropped_filename = f"{filename}_cropped_expanded_{i}_{names[int(classes[i])]}.jpg"
        # Save the cropped and expanded image
        cv2.imwrite(os.path.join(save_dir, cropped_filename), cropped_image)
        print("saved "+os.path.join(save_dir, cropped_filename))
        output_file=os.path.join(save_dir, cropped_filename)
        smoke_dir=detect_smoke(output_file)
        smoke_status=False
        images.append(output_file)
       
        if os.path.exists(os.path.join(smoke_dir,'crops')):
            smoke_status=True
            smoke_count +=1
        if names[int(classes[i])]=='motorcycle' or names[int(classes[i])]=='scooter':
            two_wheelers.append(output_file)
            print("Vehicle is a two wheeler. Implement helmet detection")
            helmet_detected_folder= detect_helmet(output_file)
            two_wheeler_crop=os.path.join(helmet_detected_folder,"crops")
            helmet_status=False
            if os.path.exists(two_wheeler_crop):
                # check if with helmet
                without_helmet=os.path.join(two_wheeler_crop,'/Without Helmet')
                with_helmet=os.path.join(two_wheeler_crop,'/With Helmet')
                if os.path.exists(without_helmet):
                    #number plate detection
                    print("Number plate")
                    without_helmet_count += 1
                    helmet_status = False
                    number_plate_files=detect_numberplate(output_file)
                    vehicle_identification_numbers= perform_OCR(number_plate_files)
                    print("Vehicle identification numbers: ", vehicle_identification_numbers)
                elif os.path.exists(with_helmet):
                    with_helmet_count += 1
                elif smoke_status:
                    number_plate_files=detect_numberplate(output_file)
                    vehicle_identification_numbers= perform_OCR(number_plate_files)
                    print("Vehicle identification numbers: ", vehicle_identification_numbers)
        else:
            other_vehicles.append(output_file)
            if smoke_status:
                number_plate_files=detect_numberplate(output_file)
                vehicle_identification_numbers= perform_OCR(number_plate_files)
                print("Vehicle identification numbers: ", vehicle_identification_numbers)
        two_wheelers_count=len(two_wheelers)
        other_vehicles_count=len(other_vehicles)
    return {"Two wheelers count ": two_wheelers_count,"Other vehicles count": other_vehicles_count,"Vehicles emitting smoke": smoke_count,"Vehicles with helmet violation": without_helmet_count}
        
        


# Example usage
# image_path = "/inputs/img.jpeg"
save_dir = "/results/"
expansion_factor = 0.2  # Adjust expansion factor as needed

# Call the function to detect vehicles, save cropped images and detect violation
def detect_vehicle(request):
    if request.method=='POST':
        uploaded_file = request.FILES['file']
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        dict_counts=crop_image_detect_violations(img, model, save_dir, 0.1)
        #print(dict_counts)  # Uncommented for debugging purposes
        return HttpResponse(json.dumps(dict_counts))
    else:
        return HttpResponse("Invalid request method. Please use POST.")
