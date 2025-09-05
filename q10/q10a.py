import cv2, numpy as np, matplotlib.pyplot as plt

IMG = "sapphire.jpg"

bgr = cv2.imread(IMG)
if bgr is None: raise FileNotFoundError(IMG)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

bgr_blur = cv2.GaussianBlur(bgr, (5,5), 0)
hsv = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2HSV)

lower = np.array([90, 40, 30], dtype=np.uint8)
upper = np.array([140,255,255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

overlay = rgb.copy()
overlay[mask==0] = (overlay[mask==0]*0.35).astype(np.uint8)  

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Original"); plt.axis("off"); plt.imshow(rgb)
plt.subplot(1,2,2); plt.title("Binary Mask"); plt.axis("off"); plt.imshow(mask, cmap="gray")
plt.tight_layout(); plt.show()

cv2.imwrite("q10a_mask.png", mask)
cv2.imwrite("q10a_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
