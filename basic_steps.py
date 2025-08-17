import cv2

# Resmi yükle
img = cv2.imread("1.jpg")

# Resmi göster
cv2.imshow("Goruntu", img) 
cv2.waitKey(0)  # Bir tuşa basılana kadar bekle
cv2.destroyAllWindows()

print(img.shape) #Resmin boyutlarını gösterir 

"""(H, W) → gri tonlu görüntü

(H, W, 3) → renkli RGB görüntü

(H, W, 4) → bazen şeffaflık (alpha kanalı) da olabilir"""

# Gri tonlamaya çevir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Göster
cv2.imshow("Orjinal", img)
cv2.imshow("Gri", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(gray.shape)

img = cv2.imread("1.jpg", 0)  # 0 parametresi → direkt gri okur

# Eşikleme
_, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Göster
cv2.imshow("Gri", img)
cv2.imshow("Siyah-Beyaz", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive Threshold
thresh_adaptive = cv2.adaptiveThreshold(
    img, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, 
    11, 2
)

# Göster
cv2.imshow("Orijinal Gri", img)
cv2.imshow("Adaptive Threshold", thresh_adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gri görüntüyü yükle
img = cv2.imread("1.jpg", 0)

# Canny Edge Detection
edges = cv2.Canny(img, 100, 200)

# Göster
cv2.imshow("Orijinal Gri", img)
cv2.imshow("Kenarlar", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()