import cv2
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


# def get_state_dict(self, *args, **kwargs):
#     kwargs.pop("check_hash")
#     return load_state_dict_from_url(self.url, *args, **kwargs)
# WeightsEnum.get_state_dict = get_state_dict

class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes, num_coordinates=4):
        super(CustomEfficientNetB0, self).__init__()
        efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model = efficientnet_b0(weights="DEFAULT")
        in_features = self.model.classifier[-1].in_features

        # Altering last layer's output size
        self.model.classifier[1] = nn.Linear(1280, 1280, bias=True)

        self.classifier_head = nn.Linear(in_features, num_classes)
        self.regressor_head = nn.Linear(in_features, num_coordinates)

    def forward(self, x):
        y = self.model(x)

        class_logits = self.classifier_head(y)
        bbox_regression = self.regressor_head(y)
        return class_logits, bbox_regression

# Load pre-trained model
def load_model(model_path):
    NUM_CLASSES = 6

    model = CustomEfficientNetB0(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path), strict=False)
    return model


# Object detection and classification pipeline
def predict_on_webcam(model, labels, transform):
    cap = cv2.VideoCapture(0)  # Open the webcam

    while True:
        success, img = cap.read()

        # Preprocess the img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            model.eval()
            class_logits, bbox_regression = model(img_tensor)

        _, predicted_class = torch.max(class_logits, 1)
        predicted_label = labels[predicted_class.item()]

        # Get predicted bounding box coordinates
        predicted_bboxes = bbox_regression.squeeze().tolist()

        # Draw bounding box on the img
        bbox = [int(coord) for coord in predicted_bboxes]
        print(predicted_label)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, predicted_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Prediction', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()