import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('/home/mathvivas/Documents/AI/road.mp4')

my_file = open('classes.txt', 'r')
data = my_file.read()
class_list = data.split('\n')

count = 0

tracker1 = Tracker()
tracker2 = Tracker()
tracker3 = Tracker()

cy1 = 184   # First line
cy2 = 209   # Second line
offset = 8

upcar = {}
downcar = {}
countercarup = []
countercardown = []
upbus = {}
downbus = {}
counterbusup = []
counterbusdown = []
downtruck = {}
counterdowntruck = []

while True:
    success, frame = cap.read()
    if not success: break
    count += 1
    if count % 3 != 0: continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype('float')

    list_car = []
    list_bus = []
    list_truck = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list_car.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list_bus.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list_truck.append([x1, y1, x2, y2])

    bbox_idx_car = tracker1.update(list_car)
    bbox_idx_bus = tracker2.update(list_bus)

    #===================CAR===================#
    for bbox_car in bbox_idx_car:
        x3, y3, x4, y4, id_car = bbox_car
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2

        #===================CARUP===================#
        if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
            upcar[id_car] = (cx3, cy3)
        if id_car in upcar:
            if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
                cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id_car}', (x3, y3), 1, 1)
                if countercarup.count(id_car) == 0:
                    countercarup.append(id_car)
        
        #===================CARDOWN===================#
        if cy2 < (cy3 + offset) and cy2 > (cy3 - offset):
            downcar[id_car] = (cx3, cy3)
        if id_car in downcar:
            if cy1 < (cy3 + offset) and cy1 > (cy3 - offset):
                cv2.circle(frame, (cx3, cy3), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id_car}', (x3, y3), 1, 1)
                if countercardown.count(id_car) == 0:
                    countercardown.append(id_car)

    #===================BUS===================#
    for bbox_bus in bbox_idx_bus:
        x5, y5, x6, y6, id_bus = bbox_bus
        cx4 = int(x5 + x6) // 2
        cy4 = int(y5 + y6) // 2

        #===================UPBUS===================#
        if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
            upbus[id_bus] = (cx4, cy4)
        if id_bus in upbus:
            if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
                cv2.circle(frame, (cx4, cy4), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id_bus}', (x4, y4), 1, 1)
                if counterbusup.count(id_bus) == 0:
                    counterbusup.append(id_bus)

        #===================DOWNBUS===================#
        if cy2 < (cy4 + offset) and cy2 > (cy4 - offset):
            downbus[id_bus] = (cx4, cy4)
        if id_bus in downbus:
            if cy1 < (cy4 + offset) and cy1 > (cy4 - offset):
                cv2.circle(frame, (cx4, cy4), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x5, y5), (x6, y6), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id_bus}', (x4, y4), 1, 1)
                if counterbusdown.count(id_bus) == 0:
                    counterbusdown.append(id_bus)

    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)
    carup = len(countercarup)
    cardown = len(countercardown)
    busup = len(counterbusup)
    busdown = len(counterbusdown)
    cvzone.putTextRect(frame, f'Upcar: {carup}', (800, 20), 2, 2)
    cvzone.putTextRect(frame, f'Downcar: {cardown}', (800, 60), 2, 2)
    cvzone.putTextRect(frame, f'Upbus: {busup}', (800, 100), 2, 2)
    cvzone.putTextRect(frame, f'Downbus: {busdown}', (800, 140), 2, 2)

    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()