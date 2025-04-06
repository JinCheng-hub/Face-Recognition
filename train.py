import os
import cv2
import numpy as np
import argparse


def train(data):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )  # 載入人臉追蹤模型
    recog = cv2.face.LBPHFaceRecognizer_create()  # 啟用訓練人臉模型方法
    faces = []  # 儲存人臉位置大小的串列
    ids = []  # 記錄該人臉 id 的串列

    for i, name in enumerate(os.listdir(data)):
        for file in os.listdir(f"{data}/{name}"):
            img = cv2.imread(f"{data}/{name}/{file}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = np.array(gray, "uint8")
            face = detector.detectMultiScale(gray)
            for x, y, w, h in face:
                faces.append(img_np[y : y + h, x : x + w])
                ids.append(i)

    print("training...")  # 提示開始訓練
    recog.train(faces, np.array(ids))  # 開始訓練
    recog.save("face.yml")  # 訓練完成儲存為 face.yml
    print("ok!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="訓練資料", required=True)
    args = parser.parse_args()
    train(args.name, args.limit)


if __name__ == "__main__":
    main()
