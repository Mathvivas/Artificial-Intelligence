import cv2

# Loading a pre-trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                    'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success: break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the region of the face
        face = frame[y:y+h, x:x+w]

        # Applying Gaussian Blur to the face
        blurred_face = cv2.GaussianBlur(face, (99, 99), 20)

        # Place the blurred face into the original frame
        frame[y:y+h, x:x+w] = blurred_face

    # Display the result
    cv2.imshow('Blurred Faces', frame)

    # Exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cv2.destroyAllWindows()
cap.release()