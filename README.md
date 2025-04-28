## Road Safety Enforcement System 🚦

Cities are becoming increasingly crowded with vehicles, a trend fueled by population growth
and rising living standards. This surge in traffic has led to two major problems: rising air
pollution and a growing number of road accidents. Studies have shown that vehicle emissions
are a major culprit behind air pollution in urban areas, particularly in developing countries.
These emissions not only contribute to a decline in environmental health but also pose a
serious threat to human health, causing respiratory problems and even premature deaths.
Unfortunately, current air quality monitoring systems and traffic management solutions are
often fragmented and struggle to keep pace with the rapid growth of traffic. Similarly, the
number of road accidents, especially involving two-wheeler riders, is on the rise globally.
Traditional methods of traffic monitoring and enforcement are becoming increasingly
laborious and time-consuming as the number of vehicles continues to climb. This situation
demands a more efficient and intelligent approach to managing traffic and ensuring safety on
our roads.

To achieve an integrated system, CCTV video input is processed through a vehicle detection
model, generating frames for subsequent application of Emission Detection and Helmet
Detection models. Optical Character Recognition is then applied for Number Plate
Recognition. The selection deep learning models such as YOLOv8, YOLOv9 are chosen for
their effectiveness in various detection tasks.
For robust vehicle identification, the number plate identification encompasses detection,
where the number plate is localized, and recognition. The project employs advanced deep
learning techniques such as YOLOV9, Followed by Optical Character Recognition using
TrOCR.

For more details please refer to

Review: https://openurl.ebsco.com/EPDB%3Agcd%3A5%3A23468901/detailv2?sid=ebsco%3Aplink%3Ascholar&id=ebsco%3Agcd%3A181715085&crl=c&link_origin=www.google.com

Project execution and results: https://docs.google.com/document/d/1dZpm76O65aDAsw2zVK0zwjSdUlwwnMY_/edit?usp=drive_link&ouid=115946271941509584747&rtpof=true&sd=true 



## 📁 Project Structure

```
Road-safety-enforcement/
│
├── app/                        # Django app handling frontend and backend logic
│   ├── templates/              # HTML templates (upload form, detection result)
│   ├── views.py                # Handles API routing, renders template
│   └── urls.py                 # URL mapping
│   └── violation_detection.py  # Handles vehicle detection and vioation detection logic 
│   └── helemt_detection.py     # Handles helemt detection 
│   └── smoke_detection.py      # Handles vehicle emission detection 
│   └── detect_numberplate.py   # Handles number plate detection logic   
│
├── inputs/              # Sample traffic images and videos for testing
│
├── yolov8_model/        # YOLOv8 models for helmet violation detection
│
├── yolov9_models/       # YOLOv9 models for vehicle emission detection and recognition
│
├── yolov9/              # Cloned YOLOv9 repository (already included for easier execution)
│
├── project/             # Django project settings
│
├── execution.ipynb      # Jupyter Notebook to test model inference separately
│
├── db.sqlite3           # Django default database (stores violations records)
│
├── manage.py            # Django project management script
│
└── requirements.txt     # Required Python packages
```

---

## ⚙️ Setting Up Locally

1. **Clone the Repository**

```bash
git clone https://github.com/Mokshitha4/Road-safety-enforcement.git
cd Road-safety-enforcement
```

2. **Create and Activate a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **YOLOv9 Setup**

No additional cloning needed!  
The **YOLOv9 repository** is already included inside `yolov9/`.  

5. **Run Migrations**

```bash
python manage.py migrate
```

6. **Start the Django Server**

```bash
python manage.py runserver
```

7. **Access the Web Interface**

Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## Sample outputs:
![image](https://github.com/user-attachments/assets/665f0626-b256-4a01-814f-f0e70467f2ca)
![image](https://github.com/user-attachments/assets/febf275f-51bc-49db-932d-c3ee84403c9c)
![image](https://github.com/user-attachments/assets/1d253dcf-a319-47f8-8c7d-21013e7623f1)

![image](https://github.com/user-attachments/assets/a0e64030-74b1-49b5-bd86-3417dd49968d)



