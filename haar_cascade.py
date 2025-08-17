import cv2

# Cascade yükle
car_cascade = cv2.CascadeClassifier("cars.xml")
if car_cascade.empty():
    print("Cascade yüklenemedi! Dosya yolu veya model hatalı.")
    exit()
# Videoyu aç
cap = cv2.VideoCapture("3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Griye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Araçları algıla
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Dikdörtgen çiz
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Göster
    cv2.imshow("Haar Cascade Car Detection", frame)

    # ESC ile çık
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
