import cv2
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import numpy as np


#### LOADING THE MODEL'S WEIGHTS

NUM_CLASSES = 6

loaded_model = efficientnet_b0()
loaded_model.classifier = nn.Linear(loaded_model.classifier[-1].in_features, NUM_CLASSES)
loaded_model.load_state_dict(torch.load('model.pth'), strict=False)
print(loaded_model.classifier)



#### DETECTING IN REAL-TIME

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success, img = cap.read()
    image_np = np.array(img)

    input_np = np.expand_dims(image_np, 0)
    input_tensor = torch.from_numpy(input_np)
