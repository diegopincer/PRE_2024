from ultralytics import YOLO
import cv2
import os

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # Você pode usar outros modelos como 'yolov8s.pt', 'yolov8m.pt', etc.

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(1)  # 0 geralmente se refere à webcam padrão do notebook

while True:
    # Capturar frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção
    results = model.predict(source=frame, save=False, show=False)

    # Renderizar resultados diretamente no frame
    annotated_frame = results[0].plot()

    # Exibir o frame com as detecções
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Parar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas OpenCV
cap.release()
cv2.destroyAllWindows()
