import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # which camera to use, 0 for the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

refimg = cv2.imread('refimg.jpg')

def checkface(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, refimg.copy())
        if result['verified'] == True:
            face_match = True
        else:
            face_match = False

    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                t = threading.Thread(target=checkface, args=(frame.copy(),))  # add a comma to make a tuple, even if empty
                t.start()

            except ValueError:
                print("no face detected")
        counter += 1

        if face_match:
            cv2.putText(frame, "Face Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # BGR not RGB
        else:
            cv2.putText(frame, "No Face Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("video", frame)
    
    key = cv2.waitKey(1)  # to be able to process user input
    if key == ord('q'):  # if user presses q, quit
        break

cap.release()
cv2.destroyAllWindows()
