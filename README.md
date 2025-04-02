# Real-Time Face Recognition System using FaceNet + KNN + Dlib

> Local real-time face recognition with live video feed, alignment, and identity prediction.

---

## 🟣 Project Description

This project implements a **real-time face recognition system** using:

- **FaceNet** (Keras FaceNet) for face embedding generation.
- **K-Nearest Neighbors (KNN)** classifier for identity prediction.
- **Dlib** for face detection and landmark extraction.
- **OpenCV** for video processing.

The system supports face alignment, live camera feed, automatic embedding creation, and real-time face recognition.

---

## ✨ Features

- Real-time face detection using Dlib.
- Automatic face alignment (eye-based normalization).
- Face embeddings generation using FaceNet.
- Classification with KNN.
- Embeddings persistence with `pickle`.
- Multi-threaded frame processing for performance.

---

## 🧩 Components

- **Face Detector**: Dlib frontal face detector.
- **Face Aligner**: Aligns detected faces based on eyes.
- **Embedder**: FaceNet model to extract 128-D embeddings.
- **Classifier**: KNN classifier to identify known faces.
- **Embedding Store**: Stores face embeddings and labels in `embeddings.pickle`.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Dlib
- Keras FaceNet
- Scikit-learn (KNN Classifier)
- NumPy
- Threading & Queues

---

## 🗃️ Directory Structure

```
face_recognition_system/
├── main.py
├── embeddings.pickle   # Auto-generated after first run
├── shape_predictor_68_face_landmarks.dat
├── README.md
└── requirements.txt
```

---

## ⚙️ How It Works

1. Detect faces from webcam using Dlib.
2. Align face using landmarks.
3. Extract 128-D embedding using FaceNet.
4. Predict identity using KNN classifier.
5. Display real-time bounding boxes and names.
6. Save new embeddings automatically when closing the program.

---

## 🟢 How to Run

1. Download `shape_predictor_68_face_landmarks.dat` and place it in the project directory.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
   python main.py
   ```
4. Press `q` to quit and automatically save embeddings.

---

## ✅ Notes

- First run may take longer if there are no existing embeddings.
- Adjust `n_neighbors` in KNN as per dataset size.
- The system will label unknown faces as `Unknown` if distance > 0.6.
- Make sure to gather enough face samples for each identity to improve accuracy.

---

## ✨ Future Improvements

- GUI for easier embedding management.
- Online embedding update.
- Deep metric learning classifier.
- Embedding database persistence in SQLite.

---

## ✨ Credits

Developed by Sagar Shankaran for educational and experimental use.
