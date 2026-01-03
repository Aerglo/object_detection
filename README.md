# 👁️ YOLOv8 Object Detection API

A robust RESTful API powered by **Django** and **Ultralytics YOLOv8** to perform object detection on images. Designed to be containerized with **Docker** and deployed seamlessly on **Hugging Face Spaces**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Django](https://img.shields.io/badge/Django-5.0-green?logo=django)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-purple)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

## 🚀 Features

* **Object Detection:** Identifies 80+ classes of objects (Person, Car, Dog, etc.) using the YOLOv8 Nano model.
* **API First:** Returns clean JSON responses including labels, confidence scores, and bounding box coordinates.
* **Optimized for Cloud:** Uses `opencv-python-headless` and CPU-optimized PyTorch for lightweight deployment.
* **Dockerized:** Ready for deployment on Hugging Face Spaces or any container orchestration platform.

## 🛠️ Tech Stack

* **Framework:** Django + Django REST Framework (DRF)
* **AI Model:** Ultralytics YOLOv8 (Nano version)
* **Image Processing:** Pillow & OpenCV
* **Deployment:** Docker (Non-root user configuration)

## 🔌 API Documentationش

### Detect Objects
**Endpoint:** `POST /api/detect/`

**Request Body:** `multipart/form-data`
* `image`: The image file to analyze (jpg, png).

**Response Example:**
```json
{
    "message": "Detection successful",
    "count": 2,
    "detections": [
        {
            "label": "person",
            "confidence": 0.88,
            "box": {
                "x1": 150.5,
                "y1": 50.2,
                "x2": 300.0,
                "y2": 450.8
            }
        },
        {
            "label": "bicycle",
            "confidence": 0.75,
            "box": {
                "x1": 310.0,
                "y1": 200.0,
                "x2": 400.0,
                "y2": 350.0
            }
        }
    ]
}
```
📦 Installation & Local Run
Clone the repository:

```Bash

git clone [https://github.com/your-username/object-detection-api.git](https://github.com/your-username/object-detection-api.git)
cd object-detection-api
```
Create a Virtual Environment:

```Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies:

```Bash

pip install -r requirements.txt
```
Run Migrations & Server:

```Bash

python manage.py migrate
python manage.py runserver
```
🐳 Deployment (Hugging Face / Docker)
This project includes a production-ready Dockerfile configured for non-root users (required for Hugging Face Spaces).

Create a Docker Space on Hugging Face.

Upload the code.

The API will be live at https://huggingface.co/spaces/USERNAME/SPACE_NAME.

Made with ❤️ by Nima
