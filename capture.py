import os
import cv2

name = input("Name: ")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

root = os.path.join("dataset", name)
os.makedirs(root, exist_ok=True)
count = 0

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(5)
    if key == ord("c"):
        count += 1
        face = gray[y : y + h, x : x + w]
        resized_face = cv2.resize(face, (100, 100))
        cv2.imwrite(f"{root}/{count}.jpg", resized_face)
        print(f"Saved {count}.jpg")

    elif key == ord("q"):
        print(f"Saved {count} faces")
        break

cap.release()
cv2.destroyAllWindows()
