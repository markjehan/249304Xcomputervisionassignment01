import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG = "daisy.jpg"

bgr = cv2.imread(IMG)
if bgr is None:
    raise FileNotFoundError(IMG)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]

mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

x = w // 20; y = h // 20
rect = (x, y, w - 2 * x, h - 2 * y)
cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask_bin = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
).astype("uint8")

SIGMA = 8 * 0.85

blurred = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=SIGMA, sigmaY=SIGMA)

alpha = (mask_bin / 255.0)[..., None]
enhanced = (alpha * rgb + (1 - alpha) * blurred).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.title("Original"); plt.axis("off"); plt.imshow(rgb)
plt.subplot(1, 2, 2); plt.title(f"Enhanced (σ={SIGMA:.2f})"); plt.axis("off"); plt.imshow(enhanced)
plt.tight_layout(); plt.show()

cv2.imwrite("q8b_enhanced_blur_bg_15pct.png", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
