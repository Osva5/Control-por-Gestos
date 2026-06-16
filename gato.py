import cv2
import mediapipe as mp
import math

# Configuracion de MediaPipe (deteccion de manos)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,   #Confianza minima para detectar las manos 
    min_tracking_confidence=0.8     #Confianza minima para rastrear las manos 
)
mp_draw = mp.solutions.drawing_utils

#Ruta del video y configuracion de la camara
video_path = 'Recursos/gato.mp4'
cap_cat = None
cap = cv2.VideoCapture(0)   #Abrir la camara web
video_encendido = False

#Funcion para verificarf si dos puntos estan cerca en el espacio de la mano 
def puntos_cerca(p1, p2, umbral=0.1):
    distancia = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return distancia < umbral

#Verificar si la mano esta en un puño cerrado
def verificar_punio(lm):
    puntas = [8, 12, 16, 20] #Puntas de los dedos (indice, medio, anular, meñique)
    base_palma = lm[0]  #Muñeca
    cerrados = [puntos_cerca(lm[p], base_palma, 0.25) for p in puntas] 
    return all(cerrados)

#Verificar si la palma esta completamente abierta
def verificar_palma_abierta(lm):
    puntas_base = [(8, 6), (12, 18), (16, 14), (20, 18)]    #Puntas y bases de los dedos 
    return all([lm[p].y < lm[b].y for p, b in puntas_base])

#Bucle principal para la detencion de gestos y control de video
while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)  #Efecto espejo para la camara
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)     #Procesar la imagen con MediaPipe

    hay_punio = False
    hay_palma = False

    #Si se detecta manos, procesamos los gestos 
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)    #Dibujar puntos
            lm = hand_landmarks.landmark

            #Verificar si es un puño o una palma abierta
            if verificar_punio(lm):
                hay_punio = True
            elif verificar_palma_abierta(lm):
                hay_palma = True

    #Control del video: Iniciar si detecta puño, detener si dectar palma
    if hay_punio and not video_encendido:
        cap_cat = cv2.VideoCapture(video_path)  #Abrir video 
        if cap_cat.isOpened():
            video_encendido = True
            print("¡Puño detectado! Gato ON.")

    elif hay_palma and video_encendido:
        video_encendido = False    #Detener video
        if cap_cat: cap_cat.release()
        cv2.destroyWindow("Gato")
        print("Palma detectada: Gato OFF.")

    #Mostrar el video si esta encendido
    if video_encendido and cap_cat is not None:
        ret_v, frame_cat = cap_cat.read()
        if not ret_v:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0) #Reiniciar video
            ret_v, frame_cat = cap_cat.read()

        if ret_v:
            frame_cat = cv2.resize(frame_cat, (400, 400))   #Redimensionar video 
            cv2.imshow("Gato", frame_cat)   #Mostrar video

    cv2.imshow("Camara Principal", frame)   #Mostrar camara en vivo

    #Salir del bucle si se preciona 'Esc'
    if cv2.waitKey(1) & 0xFF == 27: break

#Liberar recursos
cap.release()
if cap_cat: cap_cat.release()
cv2.destroyAllWindows()