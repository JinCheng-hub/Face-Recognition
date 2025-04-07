import cv2
import argparse
import pickle

def load_model(model):
    try:
        with open(model, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Loading model error: {e}")
        return None


def detect(source, model, threshold):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 1080, 1920)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            if w < 50 and h < 50:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            face = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (100, 100))
            features = resized_face.flatten().reshape(1, -1)
            prediction = model.predict(features)
            conf = model.predict_proba(features)[0]
            if max(conf) < threshold:
                continue
            cv2.putText(frame, f'{prediction}: {max(conf):.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(5) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="辨識資料", default=0)
    parser.add_argument("--model", help="辨識模型", required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold (0-1)")
    args = parser.parse_args()
    try:
        args.source = int(args.source)
    except ValueError:
        pass
    model = load_model(args.model)
    detect(args.source, model, args.threshold)


if __name__ == "__main__":
    main()
