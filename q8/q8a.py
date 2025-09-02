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

x = w // 20
y = h // 20
rect = (x, y, w - 2 * x, h - 2 * y)

cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

is_fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
mask_bin = np.where(is_fg, 255, 0).astype("uint8")


foreground = cv2.bitwise_and(rgb, rgb, mask=mask_bin)
background = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(mask_bin))

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.title("Original"); plt.axis("off"); plt.imshow(rgb)
plt.subplot(2, 2, 2); plt.title("Segmentation Mask"); plt.axis("off"); plt.imshow(mask_bin, cmap="gray")
plt.subplot(2, 2, 3); plt.title("Foreground"); plt.axis("off"); plt.imshow(foreground)
plt.subplot(2, 2, 4); plt.title("Background"); plt.axis("off"); plt.imshow(background)
plt.tight_layout(); plt.show()

cv2.imwrite("q8a_mask.png", mask_bin)
cv2.imwrite("q8a_foreground.png", cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR))
cv2.imwrite("q8a_background.png", cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
