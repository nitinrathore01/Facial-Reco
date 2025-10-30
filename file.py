import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import time

dataset_dir = "C:\\Users\\nitin\\Downloads\\Faces"          # Folder containing subfolders for each person
model_path = "face_rec_model.pkl" # Where the SVM will be saved
device = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold = 0.85              # Probability threshold for "Unknown"

# Normalizer for FaceNet embeddings
l2_normalizer = Normalizer('l2')

# Initialize MTCNN & FaceNet
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------------------------
# Function: Train model
# --------------------------
def train_model():
    print("Training model...")
    X, y = [], []

    # Loop over each person folder
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                continue

            face = mtcnn(img)
            if face is None:
                continue
            face = face.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = resnet(face).cpu().numpy()
            embedding = l2_normalizer.transform(embedding)

            X.append(embedding[0])
            y.append(person_name)  # Use folder name as label

    if len(X) == 0:
        raise ValueError("No faces found in dataset! Check your dataset folder.")

    X = np.array(X)
    y = np.array(y)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump({'clf': clf}, f)

    print("Training complete. Model saved.")
    return clf

# --------------------------
# Load or Train model
# --------------------------
if os.path.exists(model_path):
    print("Loading trained model...")
    data = pickle.load(open(model_path, 'rb'))
    clf = data['clf']
else:
    clf = train_model()

class_names = clf.classes_  # Now contains actual names

# --------------------------
# Real-time Recognition
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

time.sleep(2)  # Camera warm-up
print("Starting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (160, 160))
            face_tensor = torch.tensor(face_img / 255.0).permute(2,0,1).unsqueeze(0).float()

            with torch.no_grad():
                embedding = resnet(face_tensor).numpy()
            embedding = l2_normalizer.transform(embedding)

            probs = clf.predict_proba(embedding)
            best_idx = np.argmax(probs)
            best_prob = probs[0][best_idx]

            name = class_names[best_idx] if best_prob > threshold else "Unknown"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} {best_prob:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    frame_small = cv2.resize(frame, (640,480))
    cv2.imshow("Face Recognition", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
