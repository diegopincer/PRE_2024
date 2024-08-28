import cv2
from ultralytics import YOLO

# Modèle pour detecter des objets sur une image fourni

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolo_datasets/yolov8n.pt')  # on peut choisir d'autres modèles comme 'yolov8s.pt', m, l et x

# Realizar a detecção
results = model.predict(source='image/bus.jpg', save=True, conf=0.4, show=False,)  # save=False e show=False para não salvar ou mostrar a imagem automaticamente

# Renderizar resultados diretamente na imagem
annotated_img = results[0].plot()  # Anotar a imagem com as detecções

# Exibir a imagem
cv2.imshow('YOLOv8 Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()