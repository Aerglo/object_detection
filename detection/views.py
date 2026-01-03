from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from ultralytics import YOLO
from PIL import Image
import numpy as np




try:
    model = YOLO('yolov8n.pt') 
except Exception as e:
    print(f"Error loading model: {e}")

class ObjectDetectionView(APIView):
    
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']

        try:
            
            img = Image.open(image_file)

            
            
            results = model(img, conf=0.4) 

            
            
            detections = []
            
            
            result = results[0] 
            
            for box in result.boxes:
                
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                
                confidence = float(box.conf[0])
                
                
                cls_id = int(box.cls[0])
                
                
                label = result.names[cls_id]

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2), 
                    "box": {
                        "x1": x1, 
                        "y1": y1, 
                        "x2": x2, 
                        "y2": y2  
                    }
                })

            
            return Response({
                "message": "Detection successful",
                "count": len(detections),
                "detections": detections
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)