from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

# Two models
SPEED_SIGNAL_MODEL_PATH = "best.pt"      # Green/Red lights + speed limits + stop
GENERAL_MODEL_PATH      = "best (1).pt"  # All other traffic signs

# Prepare folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models
speed_signal_model = YOLO(SPEED_SIGNAL_MODEL_PATH)
general_model      = YOLO(GENERAL_MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    combined_path = None
    detected_text = []

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return "No file selected", 400

        # 1) Save upload
        in_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(in_path)

        # 2) Read original image
        img = cv2.imread(in_path)

        # 3) Run both models (no auto-save)
        r1 = speed_signal_model.predict(source=in_path, conf=0.20, save=False)[0]
        r2 = general_model.predict(source=in_path, conf=0.20, save=False)[0]

        # 4) Draw detections and extract text labels
        def draw_boxes_and_extract_text(r, names, color):
            nonlocal detected_text
            # r.boxes.xyxy:   (n,4) array of coordinates
            # r.boxes.cls:    (n,) array of class indices
            for xyxy, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                label = names[int(cls_id)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )
                detected_text.append(label)  # Store detected text

        # Green boxes for speed/signal
        draw_boxes_and_extract_text(r1, speed_signal_model.names, (0, 255, 0))
        # Orange boxes for general signs
        draw_boxes_and_extract_text(r2, general_model.names, (0, 140, 255))

        # Print detected text in terminal
        print("Detected Text:", detected_text)

        # 5) Save combined image
        combined_path = os.path.join(OUTPUT_FOLDER, file.filename)
        cv2.imwrite(combined_path, img)

    return render_template("index.html", result_image=combined_path, detected_text=detected_text)


@app.route("/static/outputs/<filename>")
def show_result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)










