import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cvzone
import numpy as np

model = YOLO('yolov8s.pt')
names = model.model.names
print(names)

# # Using only Cell Phone (67) and Toothbrush (79)
# filtered_names = {key: names[key] for key in [67, 79]}

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(f'MOUSE --> {point}')

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
# w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))

y_line = 251

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1024, 576))
    if not success: 
        break
    else:
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator
            annotator = Annotator(frame, line_width=2)

            # print('Cell Phone or Toothbrush')
            # print(f'classes --> {clss}')
            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Getting the center of the object
                center_x = int((box[0] + box[2]) // 2)
                center_y = int((box[1] + box[3]) // 2)

                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 4)

                # Checking if it is a Cell Phone (67) or Toothbrush (79)
                if cls == 67 or cls == 79:
                    if center_y < y_line:
                        annotator.box_label(box, color=(0, 255, 255), label=names[int(cls)])
                        cvzone.putTextRect(frame, 'WARNING', (center_x - 2, center_y - 2), 1, 1)
            

    line = cv2.line(frame, (0, y_line), (1024, y_line), (0, 255, 0), 2)
    cvzone.putTextRect(frame, f'{fps}', (30, 30), 2, 2)

    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
