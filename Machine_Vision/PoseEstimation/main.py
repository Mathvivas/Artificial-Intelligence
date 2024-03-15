from ultralytics import YOLO

########################################
# RUN NOTEBOOK AND 
# DOWNLOAD BEST WEIGHTS
########################################

model = YOLO('yolov8n-pose.pt')
model = YOLO('best.pt')

results = model(source='https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUYdGlnZXIgd2Fsa2luZyByZWZlcmVuY2Ug', show=True)