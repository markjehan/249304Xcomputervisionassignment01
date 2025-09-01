import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG = "einstein.png"

img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(IMG)

Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
Ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)

gx = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=Kx, borderType=cv2.BORDER_DEFAULT)
gy = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=Ky, borderType=cv2.BORDER_DEFAULT)

mag = np.sqrt(gx**2 + gy**2)

mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite("sobel_a_gx.png", cv2.convertScaleAbs(gx))
cv2.imwrite("sobel_a_gy.png", cv2.convertScaleAbs(gy))
cv2.imwrite("sobel_a_mag.png", mag_u8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(gx, cmap="gray"); plt.title("Gx (filter2D)"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(gy, cmap="gray"); plt.title("Gy (filter2D)"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(mag_u8, cmap="gray"); plt.title("Magnitude"); plt.axis("off")
plt.tight_layout(); plt.show()
