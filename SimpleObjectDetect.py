from ultralytics import YOLO
import cv2

# Modèle pour detecter des objets sur une image fourni

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('SOv2.pt')  # on peut choisir d'autres modèles comme 'yolov8s.pt', m, l et x

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(1)  # 0 geralmente se refere à webcam padrão do notebook

while True:
    # Capturar frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção
    results = model.predict(source=1, save=True, conf=0.3, show=True,)

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