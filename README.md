# Control de Video por Gestos con Visión Artificial 
# Meme Gato 

---

## Introducción

Gato es una aplicación de visión artificial desarrollada en Python que utiliza OpenCV y MediaPipe para detectar gestos de la mano a través de la cámara web. El programa permite controlar la reproducción de un video (`gato.mp4`) mediante dos gestos: un **puño cerrado** inicia el video, mientras que una **palma abierta** lo detiene.

La aplicación dibuja en tiempo real los puntos de referencia (landmarks) de las manos detectadas sobre el feed de la cámara, demostrando el uso de modelos de machine learning para la interpretación de gestos en tiempo real.

El objetivo principal es explorar el pipeline de visión artificial: captura de video, procesamiento con MediaPipe, clasificación de gestos basada en geometría de puntos y control de flujo multimodales (cámara + reproducción de video).

---

## Entorno y requisitos

Para ejecutar correctamente este programa se requiere lo siguiente:

- **Sistema operativo:** Windows 10/11, Linux o macOS.
- **Python:** Versión 3.8 o superior.
- **Cámara web:** Integrada o externa, accesible desde OpenCV.
- **Dependencias:** Las siguientes librerías de Python deben estar instaladas:
  - `opencv-python` — captura y procesamiento de video.
  - `mediapipe` — detección y seguimiento de manos.
  - `math` — cálculos de distancia euclidiana (librería estándar).
- **Archivo de video:** El archivo `gato.mp4` debe estar en el mismo directorio que `gato.py`.

Instalación de dependencias:

```bash
pip install opencv-python mediapipe
```

---

## Ejecución

Para ejecutar el programa siga estos pasos:

1. Asegúrese de tener una cámara web conectada y funcionando.
2. Coloque el archivo `gato.mp4` en el mismo directorio que `gato.py`.
3. Abra una terminal en el directorio del proyecto.
4. Ejecute el siguiente comando:

   ```
   python gato.py
   ```

5. Aparecerán dos ventanas:
   - **Camara Principal:** Muestra el feed en vivo con los puntos de referencia de la mano dibujados.
   - **Gato:** Muestra el video del gato (solo cuando se detecta un puño).
6. Para salir del programa, presione la tecla `Esc`.

---

## Código y funcionamiento

### Configuración de MediaPipe

```python
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils
```

Se inicializa el modelo de detección de manos con una confianza mínima del 80%. Se configuran un máximo de 2 manos para permitir detección de gestos simultáneos si es necesario.

### Captura de cámara y carga del video

```python
video_path = 'gato.mp4'
cap = cv2.VideoCapture(0)
video_encendido = False
```

Se abre la cámara web (índice 0) y se define la ruta del archivo de video del gato. La variable `video_encendido` controla el estado de reproducción.

### Función puntos_cerca

```python
def puntos_cerca(p1, p2, umbral=0.1):
    distancia = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return distancia < umbral
```

Calcula la distancia euclidiana entre dos puntos normalizados (coordenadas `x`, `y` del landmark) y retorna `True` si están por debajo del umbral. Se utiliza para determinar si los dedos están cerca de la palma.

### Función verificar_punio

```python
def verificar_punio(lm):
    puntas = [8, 12, 16, 20]
    base_palma = lm[0]
    cerrados = [puntos_cerca(lm[p], base_palma, 0.25) for p in puntas]
    return all(cerrados)
```

Verifica si la mano forma un puño cerrado. Toma las puntas de los cuatro dedos (índice, medio, anular, meñique) y comprueba si todas están cerca de la muñeca (landmark 0). Si los cuatro dedos están dentro del umbral, se considera un puño.

### Función verificar_palma_abierta

```python
def verificar_palma_abierta(lm):
    puntas_base = [(8, 6), (12, 18), (16, 14), (20, 18)]
    return all([lm[p].y < lm[b].y for p, b in puntas_base])
```

Verifica si la mano está completamente abierta. Compara la coordenada `y` de cada punta del dedo con la de su base correspondiente. Si todas las puntas están por encima de sus bases (menor `y`), la palma está abierta.

### Bucle principal

```python
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
```

El bucle principal captura cada fotograma de la cámara, lo invierte horizontalmente (efecto espejo), lo convierte a RGB (formato requerido por MediaPipe) y procesa la imagen para detectar manos.

```python
if resultado.multi_hand_landmarks:
    for hand_landmarks in resultado.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = hand_landmarks.landmark
        if verificar_punio(lm): hay_punio = True
        elif verificar_palma_abierta(lm): hay_palma = True
```

Cuando se detectan manos, se dibujan los 21 landmarks y sus conexiones sobre el fotograma. Luego se evalúa el gesto: si es puño o palma abierta.

```python
if hay_punio and not video_encendido:
    cap_cat = cv2.VideoCapture(video_path)
    if cap_cat.isOpened():
        video_encendido = True
        print("¡Puño detectado! Gato ON.")
elif hay_palma and video_encendido:
    video_encendido = False
    if cap_cat: cap_cat.release()
    cv2.destroyWindow("Gato")
    print("Palma detectada: Gato OFF.")
```

La lógica de control: un puño inicia la reproducción del video del gato; una palma abierta lo detiene y cierra la ventana del video.

```python
if video_encendido and cap_cat is not None:
    ret_v, frame_cat = cap_cat.read()
    if not ret_v:
        cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_v, frame_cat = cap_cat.read()
    if ret_v:
        frame_cat = cv2.resize(frame_cat, (400, 400))
        cv2.imshow("Gato", frame_cat)
```

Si el video está encendido, se lee y muestra cada fotograma redimensionado a 400×400 píxeles. Cuando el video termina, se reinicia automáticamente desde el primer fotograma.

---


La principal ventaja de utilizar MediaPipe es que abstrae la complejidad del modelo de detección de manos, proporcionando 21 puntos de referencia precisos que permiten clasificar gestos con pocas líneas de código. Esto acelera el desarrollo y mejora la precisión en comparación con métodos tradicionales de visión artificial.
