import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 nano pretrained model for segmentation
model = YOLO('yolov8n-seg.pt')
 
# Open the video file
video_path = "cars.MOV"
cap = cv2.VideoCapture(1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        start = time.perf_counter()
        # Run YOLOv8 inference on the frame
        results = model(frame)
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        #visualize the results on the frame
        annotated_frame = results[0].plot()

        # display the annotated frame
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("YOLOv8_Inference", annotated_frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()





