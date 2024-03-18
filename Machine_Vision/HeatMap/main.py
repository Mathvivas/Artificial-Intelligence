from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('vehicles.mp4')
assert cap.isOpened(), 'Error reading video file'
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter('heatmap_output.avi',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (w, h))

# Init heatmap
heatmap_obj = heatmap.Heatmap()

# Resize parameters
new_width = 640
new_height = 480

# Set heatmap arguments
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=new_width,  # Set the width after resizing
                     imh=new_height,  # Set the height after resizing
                     view_img=True,
                     shape='circle')

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print('Video frame is empty or video processing has been successfully completed.')
        break
    frame = cv2.resize(frame, (new_width, new_height))
    tracks = model.track(frame, persist=True, show=False)

    frame = heatmap_obj.generate_heatmap(frame, tracks)
    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()