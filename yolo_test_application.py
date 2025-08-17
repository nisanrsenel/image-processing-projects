from ultralytics import YOLO
import cv2
import numpy as np

# YOLOv8 hazır model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("3.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("red_cars_detected.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile arabaları algıla
    results = model(frame)
    red_car_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Sadece arabalar ve %50 üzeri güven
            if model.names[cls] == "car" and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ROI: sadece araç kısmını kırmızı kontrol için al
                roi = frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Kırmızı renk aralıkları
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])

                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = mask1 + mask2
                
                # Morfolojik işlemler: küçük lekeleri temizle
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                # Kırmızı piksel oranı
                red_ratio = np.sum(mask > 0) / (roi.shape[0]*roi.shape[1])

                if red_ratio > 0.05:  # %5’ten fazla kırmızı ise say
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"Red Car {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    red_car_count += 1

    # Kırmızı araba sayısı
    cv2.putText(frame, f"Red Cars: {red_car_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("Red Car Detection YOLO", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
