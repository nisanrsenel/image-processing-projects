import cv2
import numpy as np

# Resmi yükle
img = cv2.imread("2.jpg")

# BGR → HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Kırmızı renk aralığı (HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Maske oluştur
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Morfolojik işlemler: parçaları birleştir
kernel = np.ones((5,5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)  # beyaz bölgeleri büyüt
mask = cv2.erode(mask, kernel, iterations=1)   # küçük gürültüleri sil

### Seçilen renkteki nesneleri belirleme ###

# Konturları bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Nesneleri kare içine al
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

# Sayıyı yaz         
cv2.putText(img, f"Count: {len(contours)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Göster
cv2.imshow("Kirmizi Nesneler", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Arkaplanı karartma ###

# Sadece kırmızı bölgeyi göster
red_only = cv2.bitwise_and(img, img, mask=mask)

# Konturları bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Nesneleri kare içine al
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(red_only, (x, y), (x+w, y+h), (0,255,0), 2)

# Sayıyı yaz
cv2.putText(red_only, f"Count: {len(contours)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# Göster
cv2.imshow("Sadece Kirmizi Nesneler", red_only)
cv2.waitKey(0)
cv2.destroyAllWindows()