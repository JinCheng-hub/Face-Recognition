import os
import cv2
import argparse


def capture(name, limit):
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        exit()

    root = os.path.join("dataset", name)
    os.makedirs(root, exist_ok=True)
    count = 0

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            eyes = face_detector.detectMultiScale(face, scaleFactor=1.05, minNeighbors=5)
            if len(eyes) == 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                resized_face = cv2.resize(face, (100, 100))
                cv2.imwrite(f"{root}/{count}.jpg", resized_face)
                print(f"Saved {count}.jpg")

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            print(f"Exiting...\n\nSaved {count} faces")
            break

        if count >= limit:
            print(f"Saved {count} faces")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="誰的臉", required=True)
    parser.add_argument("--limit", type=int, help="擷取上限", default=25)
    args = parser.parse_args()
    capture(args.name, args.limit)


if __name__ == "__main__":
    main()
