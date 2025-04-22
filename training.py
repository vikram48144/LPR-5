from ultralytics import YOLO

# Initialize the YOLO model (you can use a pre-trained model)
model = YOLO("yolov8n.pt")  # You can choose a different pre-trained model here

# Train the model
model.train(
    data="C:/LPR-5/data.yaml",   # Your path to data.yaml
    epochs=50,                    # Adjust epochs as per your requirement
    imgsz=640,                    # Image size
    batch=16,                     # Batch size
    project="C:/LPR-5/yolov8_training",  # Directory to save the trained model and results
    name="plate_detector"         # Name of the experiment
)

print("Training started...")
