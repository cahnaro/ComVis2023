import cv2
import os
import numpy as np

# Path dataset wajah
dataset_path = "dataset_wajah"

# Fungsi untuk membaca dataset wajah dan mengembalikan array wajah dan label
def read_dataset():
    faces = []
    labels = []
    # Loop setiap direktori di dataset
    for dir_name in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, dir_name)):
            continue
        label = int(dir_name)
        # Loop setiap gambar di direktori
        for image_name in os.listdir(os.path.join(dataset_path, dir_name)):
            image_path = os.path.join(dataset_path, dir_name, image_name)
            # Membaca gambar dan mengubah ke grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Menambahkan wajah dan label ke array
            faces.append(image)
            labels.append(label)
    return faces, labels

# Fungsi untuk membuat model pengenalan wajah dengan metode Eigenface
def create_eigenface_model(faces, labels):
    # Membuat objek face recognizer menggunakan metode Eigenface
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    # Melatih model dengan wajah dan label yang telah dibaca
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

# Fungsi untuk membuat model pengenalan wajah dengan metode Fisherface
def create_fisherface_model(faces, labels):
    # Membuat objek face recognizer menggunakan metode Fisherface
    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    # Melatih model dengan wajah dan label yang telah dibaca
    face_recognizer.train(faces, np.array(labels))
    return face_recognizer

# Fungsi untuk memprediksi label wajah menggunakan model pengenalan wajah
def predict_face_label(face_recognizer, image):
    # Mengubah gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Mendeteksi wajah di gambar
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Membuat ROI dari wajah dan mengubah ukuran menjadi 200x200
    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (200, 200))
    # Memrediksi label wajah menggunakan model pengenalan wajah
    label, confidence = face_recognizer.predict(face_roi)
    return label

# Membaca dataset wajah dan mengembalikan array wajah dan label
faces, labels = read_dataset()

# Membuat model pengenalan wajah dengan metode Eigenface
eigenface_model = create_eigenface_model(faces, labels)

# Membuat model pengenalan wajah dengan metode Fisherface
fisherface_model = create_fisherface_model(faces, labels)

# Membaca gambar untuk diprediksi
