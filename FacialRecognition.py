import cv2
import face_recognition
import numpy as np
import os
import base64
from datetime import datetime
import csv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Connexion MongoDB
uri = "mongodb+srv://yassinejada01:Raja1949@cluster0.zjbbizf.mongodb.net/"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['face_recognition']
employees_col = db['employees']


# Chargement des images et des noms des employés depuis MongoDB
def load_employee_images():
    global images, classNames
    images = []
    classNames = []
    for record in employees_col.find():
        img_data = base64.b64decode(record['image'])
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        images.append(img)
        classNames.append(record['name'])
    print("Liste des employés mise à jour :", classNames)

load_employee_images()

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encode = encodings[0]
            encodeList.append(encode)
        else:
            print("Aucun visage détecté dans l'image.")
    return encodeList

def add_new_employee(name):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Capture New Employee Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            _, buffer = cv2.imencode('.png', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            employees_col.insert_one({'name': name, 'image': img_base64})
            break
    video_capture.release()
    cv2.destroyAllWindows()
    load_employee_images()
    print(f"Employé {name} ajouté avec succès dans la base de données.")

def log_attendance(name):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, dtString])
    print(f"Pointage enregistré pour {name} à {dtString}")

def facial_recognition():
    encodeListKnown = findEncodings(images)
    print('Encodage terminé')
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                log_attendance(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Interface Tkinter
def start_facial_recognition():
    messagebox.showinfo("Info", "Démarrage de la reconnaissance faciale...")
    facial_recognition()

def add_employee():
    name = name_entry.get()
    if name:
        add_new_employee(name)
        messagebox.showinfo("Succès", f"Employé {name} ajouté avec succès !")
    else:
        messagebox.showwarning("Erreur", "Veuillez entrer un nom.")

window = tk.Tk()
window.title("Système de reconnaissance faciale des employés")
window.geometry("500x400")

# Interface pour ajouter un employé
add_frame = tk.Frame(window)
add_frame.pack(pady=10)
tk.Label(add_frame, text="Nom de l'employé :").grid(row=0, column=0, padx=5)
name_entry = tk.Entry(add_frame)
name_entry.grid(row=0, column=1, padx=5)
add_button = tk.Button(add_frame, text="Ajouter l'employé", command=add_employee)
add_button.grid(row=0, column=2, padx=5)

# Bouton pour démarrer la reconnaissance faciale
start_button = tk.Button(window, text="Démarrer la reconnaissance faciale", command=start_facial_recognition)
start_button.pack(pady=20)

# Bouton pour quitter
exit_button = tk.Button(window, text="Quitter", command=window.quit)
exit_button.pack(pady=10)

# Boucle principale
window.mainloop()
