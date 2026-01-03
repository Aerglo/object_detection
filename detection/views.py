from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. لود کردن مدل YOLO (یک بار در زمان اجرای سرور)
# مدل 'yolov8n.pt' نسخه نانو است که سریع‌ترین نسخه می‌باشد.
# برای دقت بیشتر می‌توانید از yolov8s.pt یا yolov8m.pt استفاده کنید.
try:
    model = YOLO('yolov8n.pt') 
except Exception as e:
    print(f"Error loading model: {e}")

class ObjectDetectionView(APIView):
    # پارسرها برای دریافت فایل (عکس) ضروری هستند
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        # 2. دریافت عکس از ریکوئست
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']

        try:
            # 3. باز کردن عکس با Pillow
            img = Image.open(image_file)

            # 4. انجام تشخیص شیء
            # conf=0.4 یعنی فقط اشیایی که بالای 40% اطمینان دارند را نشان بده
            results = model(img, conf=0.4) 

            # 5. پردازش نتایج برای ارسال به کلاینت
            # مدل YOLO نتایج پیچیده‌ای برمی‌گرداند، ما باید آن را ساده کنیم
            detections = []
            
            # معمولاً یک عکس ورودی داریم، پس results[0] را می‌گیریم
            result = results[0] 
            
            for box in result.boxes:
                # مختصات کادر (x1, y1, x2, y2)
                # مقادیر به صورت Tensor هستند، باید به float تبدیل شوند
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # میزان اطمینان (Confidence)
                confidence = float(box.conf[0])
                
                # کلاس شیء (مثلاً 0 برای انسان، 2 برای ماشین و ...)
                cls_id = int(box.cls[0])
                
                # نام کلاس (مثلاً "person")
                label = result.names[cls_id]

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2), # گرد کردن تا دو رقم اعشار
                    "box": {
                        "x1": x1, # چپ
                        "y1": y1, # بالا
                        "x2": x2, # راست
                        "y2": y2  # پایین
                    }
                })

            # 6. بازگرداندن خروجی JSON
            return Response({
                "message": "Detection successful",
                "count": len(detections),
                "detections": detections
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)