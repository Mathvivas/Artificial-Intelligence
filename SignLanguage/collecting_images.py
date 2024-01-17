import cv2
import os
import time
import uuid

IMAGES_PATH = "images/collected_images"

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15

# Collecting Images
for label in labels:
    os.system(f'mkdir {IMAGES_PATH}/' + label)
    cap = cv2.VideoCapture(0)
    print(f'Collecting images for {label}')
    time.sleep(5)
    for imgnum in range(number_imgs):
        success, img = cap.read()
        imagename = os.path.join(IMAGES_PATH, label, 
                label + '.' + f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imagename, img)
        cv2.imshow('Image', img)
        time.sleep(2)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    cap.release()