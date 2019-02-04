import cv2
import numpy as np
import test

def fromWebcam():

    print ("Opening webcam...")
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi = gray

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi = gray[y:y+h, x:x+w]

        if roi is not None:
            gray = roi
        resized_img = cv2.resize(gray, (48, 48))
        test.predict(resized_img.reshape(-1, 48, 48, 1), "Predict")

        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print ("Exiting...")
    cap.release()
    cv2.destroyAllWindows()

def fromImage(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        roi = gray[y:y+h, x:x+w]
    cv2.imshow('frame',roi)
    cv2.waitKey(0)

    if roi is not None:
        gray = roi
    resized_img = cv2.resize(gray, (48, 48))
    finalImage = resized_img.reshape(-1, 48, 48, 1)
    return finalImage, resized_img

fromWebcam()
