import os
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


def preprocess(dataset):
    features = []
    labels = []

    for name in os.listdir(dataset):
        for file in os.listdir(f"{dataset}/{name}"):
            img = cv2.imread(f"{dataset}/{name}/{file}", cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            features.append(img.flatten())
            labels.append(name)

    features = np.array(features)
    labels = np.array(labels)
    return train_test_split(features, labels, test_size=0.2)


def random_forest(data):
    X_train, X_test, y_train, y_test = data
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    print("Training the Random Forest model...")
    clf.fit(X_train, y_train)
    print("Completed!")

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    return clf


def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {name}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Model name", required=True)
    parser.add_argument("--data", help="Dataset path", default="dataset")
    args = parser.parse_args()

    data = preprocess(args.data)
    model = random_forest(data)
    save_model(model, args.name)


if __name__ == "__main__":
    main()
