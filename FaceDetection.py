import cv2
import dlib
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from threading import Thread
from queue import Queue
from keras.models import load_model
from keras_facenet import FaceNet

# Initialize FaceNet model
embedder = FaceNet()

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load or initialize embeddings database
if os.path.exists('embeddings.pickle'):
    with open('embeddings.pickle', 'rb') as f:
        data = pickle.load(f)
    known_embeddings = data['embeddings']
    known_labels = data['labels']
else:
    known_embeddings = []
    known_labels = []

# Initialize KNN classifier
if known_embeddings:
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(known_embeddings, known_labels)
else:
    knn = None

# Function to align face
def align_face(image, rect):
    shape = predictor(image, rect)
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype='float32')
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    desired_right_eye_x = 1.0 - 0.35
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - 0.35) * 256
    scale = desired_dist / dist
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tX = 256 * 0.5
    tY = 256 * 0.35
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    output = cv2.warpAffine(image, M, (256, 256), flags=cv2.INTER_CUBIC)
    return output

# Function to process frames
def process_frame(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            aligned_face = align_face(frame, rect)
            aligned_face = cv2.resize(aligned_face, (160, 160))
            aligned_face = np.expand_dims(aligned_face, axis=0)
            embedding = embedder.embeddings(aligned_face)[0]
            if knn:
                distances, indices = knn.kneighbors([embedding])
                if distances[0][0] < 0.6:
                    name = knn.predict([embedding])[0]
                else:
                    name = 'Unknown'
            else:
                name = 'Unknown'
            result_queue.put((x, y, w, h, name))
        frame_queue.task_done()

# Start video capture
cap = cv2.VideoCapture(0)
frame_queue = Queue()
result_queue = Queue()
thread = Thread(target=process_frame, args=(frame_queue, result_queue))
thread.daemon = True
thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_queue.put(frame)
    while not result_queue.empty():
        (x, y, w, h, name) = result_queue.get()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
frame_queue.put(None)
frame_queue.join()
cap.release()
cv2.destroyAllWindows()

# Save embeddings database
with open('embeddings.pickle', 'wb') as f:
    data = {'embeddings': known_embeddings, 'labels': known_labels}
    pickle.dump(data, f)
