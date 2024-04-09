from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import cv2
from torchvision import transforms

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # Preprocess the img
    transf = transforms.ToTensor()
    img_tensor = transf(img).unsqueeze(0)

    output = model(img_tensor)
    results = Detections.from_ultralytics(output[0])

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
