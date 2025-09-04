import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG = "rice.png"

g = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if g is None:
    raise FileNotFoundError(IMG)

g_blur = cv2.GaussianBlur(g, (5, 5), 0)
_, bin_ = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

opened  = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, k3, iterations=1)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5, iterations=1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Otsu Binary"); plt.axis("off"); plt.imshow(bin_, cmap="gray")
plt.subplot(1,3,2); plt.title("Opening (3×3)"); plt.axis("off"); plt.imshow(opened, cmap="gray")
plt.subplot(1,3,3); plt.title("Opening + Closing"); plt.axis("off"); plt.imshow(cleaned, cmap="gray")
plt.tight_layout(); plt.show()

cv2.imwrite("q9d_cleaned_binary.png", cleaned)
