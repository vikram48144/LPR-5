import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import os
from datetime import datetime
import pandas as pd

# Load YOLOv8 trained model
model = YOLO("C:/LPR-5/yolov8_training/plate_detector/weights/best.pt")

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], verbose=False)

# Open webcam
cap = cv2.VideoCapture(0)

# Folder to save plates
save_folder = "C:/LPR-5/plates"
os.makedirs(save_folder, exist_ok=True)

# Initialize fixed-size images for display
empty_plate_crop = np.zeros((150, 300, 3), dtype=np.uint8)  # black box
plate_crop = empty_plate_crop.copy()
plate_text_window = 255 * np.ones((100, 400, 3), dtype=np.uint8)  # white box

# Excel file to store timestamps and plate numbers
excel_path = "C:/LPR-5/plates_timestamps.xlsx"
columns = ["Plate Number", "Timestamp"]

# Load existing Excel if present, otherwise create a new DataFrame
if os.path.exists(excel_path):
    timestamps_df = pd.read_excel(excel_path)
else:
    timestamps_df = pd.DataFrame(columns=columns)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Reset text every frame
    text = ""
    plate_crop = empty_plate_crop.copy()  # reset crop if nothing detected

    # Detect and crop
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = frame[y1:y2, x1:x2]
            plate_crop = cv2.resize(plate_crop, (300, 150))  # force size for stability

            # Draw a green bounding box around the detected plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # OCR
            result = reader.readtext(plate_crop)
            if result:
                text = result[0][1]

    # Prepare white text window
    plate_text_window = 255 * np.ones((100, 400, 3), dtype=np.uint8)
    if text:
        cv2.putText(plate_text_window, text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Display all windows
    cv2.imshow("Live Webcam", frame)
    cv2.imshow("Cropped Plate", plate_crop)
    cv2.imshow("License Plate Text", plate_text_window)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and text:  # Save the cropped image when 's' is pressed
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"plate_{timestamp}.jpg")
        cv2.imwrite(filename, plate_crop)

        # Add the plate and timestamp to the DataFrame and save to Excel
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([[text, ts]], columns=columns)
        timestamps_df = pd.concat([timestamps_df, new_row], ignore_index=True)
        timestamps_df.to_excel(excel_path, index=False)  # Save back to Excel

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
