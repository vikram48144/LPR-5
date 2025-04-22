import cv2
import os
import csv

# Path to your cropped plates
cropped_folder = "C:/LPR-5/cropped_plates"  # change this to your cropped images folder
csv_file_path = "C:/LPR-5/plate_labels.csv"  # where you want to save the labels

# Get all images
images = [img for img in os.listdir(cropped_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Open CSV file to save
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "plate_text"])  # header

    for img_name in images:
        img_path = os.path.join(cropped_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load {img_path}")
            continue

        # Display the image
        cv2.imshow("Label the Plate", img)
        cv2.waitKey(1)  # needed for the window to update

        # Ask user to type plate text
        plate_text = input(f"Enter plate text for {img_name}: ")

        # Save to CSV
        writer.writerow([img_name, plate_text])

        cv2.destroyAllWindows()

print("✅ All images labeled and saved to CSV!")
