import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG = "rice.png"

g = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if g is None:
    raise FileNotFoundError(IMG)

g_blur = cv2.GaussianBlur(g, (5, 5), 0)

_, bin_ = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1); plt.title("Grayscale"); plt.axis("off"); plt.imshow(g, cmap="gray")
plt.subplot(1,2,2); plt.title("Otsu Binary"); plt.axis("off"); plt.imshow(bin_, cmap="gray")
plt.tight_layout(); plt.show()

cv2.imwrite("q9c_otsu_binary.png", bin_)
