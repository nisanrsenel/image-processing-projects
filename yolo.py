from ultralytics import YOLO
import cv2

# Küçük boyutlu önceden eğitilmiş model (COCO veri seti)
model = YOLO("yolov8n.pt")  # ilk kullanımda model indirilecek

# Video aç
cap = cv2.VideoCapture("3.mp4")

# Video özelliklerini al
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Çıkış videosu (MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("yolo_cars_detected.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi modele gönder
    results = model(frame)

    # Modelden gelen sonuçları işle
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])     
            conf = float(box.conf[0])   
            # Sadece araba (car) sınıfı ve %50 üzeri güven
            if model.names[cls] == "car" and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Car {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                count += 1

    # Sayıyı ekle
    cv2.putText(frame, f"Cars: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Kareyi videoya yaz
    out.write(frame)
    
    # Göster
    cv2.imshow("YOLO Car Detection", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
out.release()
cv2.destroyAllWindows()
