import cv2
import numpy as np

# Video dosyasını aç
cap = cv2.VideoCapture("3.mp4")

# Video özelliklerini al (genişlik, yükseklik, fps)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# Çıkış videosunu ayarla
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 formatı için
# Çıkış videosu için VideoWriter nesnesi oluştur
out = cv2.VideoWriter("cars_detected.mp4", fourcc, fps, (width, height))

# Video karelerini oku ve işleme al
while True:
    ret, frame = cap.read()
    if not ret:  # Video biterse döngüyü kır
        break

    # BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralıkları
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Maske
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morfolojik işlemler
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    min_area = 1000  # çok küçük alanları yok say
    max_area = 20000  # çok büyük alanları da yok say

    #Aspect ratio (en-boy oranı) kontrolü
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if 0.5 < aspect_ratio < 2.5:  # Araç boyutuna yakınsa
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                count += 1

    # Nesne sayısını ekrana yaz
    cv2.putText(frame, f"Cars: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Çıkış videosuna kareyi yaz
    out.write(frame)

    # Görüntüyü göster
    cv2.imshow("Red Car Detection", frame)

    # ESC tuşu (27) ile çıkış
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
