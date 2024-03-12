from ultralytics import YOLO

# Loading a pretrained model
# Run SignLanguageTraining notebook and download the best weights
model = YOLO('SignLanguage.pt')

# Run inference on the source (webcam)
results = model(source=0, show=True, conf=0.3, save=True)