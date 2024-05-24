import torch
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # on peut choisir d'autres modèles comme 'yolov8s.pt', m, l et x

# Realizar a detecção
results = model.predict(source='https://www.youtube.com/watch?v=LqBgPhyoCz0', save=True, conf=0.4, show=True, stream=True)  # save=False e show=False para não salvar ou mostrar a imagem automaticamente

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputsq
    result.show()  # display to screen
    #result.save(filename='result.jpg')  # save to disk