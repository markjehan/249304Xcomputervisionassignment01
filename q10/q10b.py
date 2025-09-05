import cv2, numpy as np, matplotlib.pyplot as plt

IMG = "sapphire.jpg"
bgr = cv2.imread(IMG)
if bgr is None: raise FileNotFoundError(IMG)

hsv = cv2.cvtColor(cv2.GaussianBlur(bgr,(5,5),0), cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([90,40,30]), np.array([140,255,255]))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

mask_before = mask.copy()

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=1)

h, w = mask.shape
ff = mask.copy()
border = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(ff, border, seedPoint=(0,0), newVal=255)    
holes = cv2.bitwise_not(ff) & mask                        
filled = cv2.bitwise_or(mask, holes)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("Mask (from a)"); plt.axis("off"); plt.imshow(mask_before, cmap="gray")
plt.subplot(1,3,2); plt.title("After Closing"); plt.axis("off"); plt.imshow(mask, cmap="gray")
plt.subplot(1,3,3); plt.title("Holes Filled"); plt.axis("off"); plt.imshow(filled, cmap="gray")
plt.tight_layout(); plt.show()

cv2.imwrite("q10b_filled.png", filled)
