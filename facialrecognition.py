import cv2
import numpy as np
import face_recognition
import os

path = 'images'
faces = []
classNames = []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    faces.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList

encodeListKnown = findEncodings(faces)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesincurr = face_recognition.face_locations(imgS)
    encodeincurr = face_recognition.face_encodings(imgS, facesincurr)

    for encodeFace, faceLoc in zip(encodeincurr, facesincurr):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            x1, y1, x2, y2 = 4*x1, 4*y1, 4*x2, 4*y2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()