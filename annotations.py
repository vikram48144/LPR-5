import cv2
import os

# ── CONFIGURE THESE PATHS ──
images_dir = r"C:\LPR-5\images"     # your folder of PNGs
labels_dir = r"C:\LPR-5\labels\k.txt"     # where .txt files will go
class_id   = 0                      # license_plate = 0
# ────────────────────────────

os.makedirs(labels_dir, exist_ok=True)

# Gather all PNGs
images = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]

for img_name in images:
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    rect = []  # will hold two points [(x1,y1),(x2,y2)]

    def on_mouse(event, x, y, flags, param):
        nonlocal rect, img
        if event == cv2.EVENT_LBUTTONDOWN:
            rect = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            rect.append((x, y))
            cv2.rectangle(img, rect[0], rect[1], (0, 255, 0), 2)
            cv2.imshow("Annotate", img)

    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", on_mouse)

    print(f"\nAnnotate: {img_name}")
    print(" Draw a box with mouse.  Press 's' to save, 'n' to skip this image.")

    while True:
        cv2.imshow("Annotate", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(rect) == 2:
                x1, y1 = rect[0]
                x2, y2 = rect[1]
                # convert to YOLO format (normalized center, width, height)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w    = abs(x2 - x1) / w
                box_h    = abs(y2 - y1) / h

                label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                print(f" → Saved annotation: {label_path}")
            else:
                print(" No box drawn, nothing saved.")
            break

        elif key == ord('n'):
            print(" → Skipped")
            break

    cv2.destroyAllWindows()
