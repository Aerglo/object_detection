
FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


WORKDIR /code


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY . /code/


RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"


RUN mkdir -p /code/media && chmod 777 /code/media


CMD ["python", "manage.py", "runserver", "0.0.0.0:7860"]