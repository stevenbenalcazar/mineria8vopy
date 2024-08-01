import cv2
import dlib
import numpy as np
from scipy.signal import find_peaks # type: ignore

# Inicializar detector de rostros y predictor de puntos faciales
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/Steven Benalcázar/Desktop/proyecto mineria/DB videos/shape_predictor_68_face_landmarks.dat')

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Listas para almacenar las señales extraídas
green_signal = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Extracción de la ROI de la frente (entre los puntos 19 y 24 de los landmarks faciales)
        forehead_points = np.array([(landmarks.part(19).x, landmarks.part(19).y),
                                    (landmarks.part(24).x, landmarks.part(24).y),
                                    (landmarks.part(24).x, landmarks.part(24).y - 20),
                                    (landmarks.part(19).x, landmarks.part(19).y - 20)])
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [forehead_points], -1, (255, 255, 255), -1)
        mean_val = cv2.mean(frame, mask=mask)[:3]
        
        # Almacenar el valor medio del canal verde de la ROI
        green_signal.append(mean_val[1])
        
        # Dibujar la ROI en el frame
        cv2.polylines(frame, [forehead_points], True, (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Conversión de la señal a un array de numpy
green_signal = np.array(green_signal)

# Normalizar la señal
green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)

# Detección de picos en la señal para calcular el ritmo cardíaco
peaks, _ = find_peaks(green_signal, distance=30)

# Calcular el ritmo cardíaco
if len(peaks) > 1:
    bpm = 60.0 / np.mean(np.diff(peaks))
    print(f'Ritmo cardíaco: {bpm:.2f} BPM')
else:
    print('No se detectaron suficientes picos para calcular el ritmo cardíaco.')
