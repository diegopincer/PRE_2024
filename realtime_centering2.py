from ultralytics import YOLO
import cv2
import socket

# Configurações do Socket UDP
UDP_IP = "172.43.100.100"  # Endereço IP do servidor
UDP_PORT = 5005       # Porta do servidor

# Criar o socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolo_datasets/yolov8n.pt')  # Você pode usar outros modelos como 'yolov8s.pt', 'yolov8m.pt', etc.

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)  # 0 geralmente se refere à webcam padrão do notebook

# Variáveis para armazenar a posição inicial
initial_position = None

while True: 
    # Capturar frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção
    results = model.predict(source=frame, save=False, show=False)

    # Obter as detecções (coordenadas das bounding boxes e classes)
    detections = results[0].boxes.data
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if class_id == 0:  # Verifica se a classe detectada é 'person' (0 é o ID para 'person')
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calcular o centro da bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Se a posição inicial não foi definida, armazene a posição atual
            if initial_position is None:
                initial_position = (center_x, center_y)

            # Calcular o delta x e delta y
            delta_x = center_x - initial_position[0]
            delta_y = center_y - initial_position[1]
            
         

            if delta_x > 220 :
                delta_y = 0
                delta_x = 0

            if delta_x < -220 :
                delta_y = 0
                delta_x = 0
            
            if delta_y > 50 :
                delta_y = 0
                delta_x = 0

            if delta_y < -50 :
                delta_y = 0
                delta_x = 0

            # Enviar os valores de delta via UDP
            message = f"{delta_x},{delta_y}".encode()
            sock.sendto(message, (UDP_IP, UDP_PORT))

            # Desenhar a bounding box ao redor da pessoa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Exibir as coordenadas da bounding box no frame
            cv2.putText(frame, f"Person: ({x1},{y1})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # Exibir o delta x e delta y no frame
            cv2.putText(frame, f"Delta: ({delta_x},{delta_y})", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir o frame com as detecções
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Parar se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas OpenCV
cap.release()
cv2.destroyAllWindows()

