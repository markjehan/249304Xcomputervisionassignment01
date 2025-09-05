import cv2, numpy as np, matplotlib.pyplot as plt

IMG = "sapphire.jpg"
bgr = cv2.imread(IMG)
if bgr is None: raise FileNotFoundError(IMG)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(cv2.GaussianBlur(bgr,(5,5),0), cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([90,40,30]), np.array([140,255,255]))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=1)

h, w = mask.shape
ff = mask.copy(); border = np.zeros((h+2,w+2), np.uint8)
cv2.floodFill(ff, border, (0,0), 255); holes = cv2.bitwise_not(ff) & mask
mask = cv2.bitwise_or(mask, holes)

count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

areas = []
for i in range(1, count):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 100:  
        areas.append((i, area))


vis = rgb.copy()
for i, area in areas:
    x,y,w,h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
    cx, cy = map(int, centroids[i])
    cv2.putText(vis, f"id {i} : {area}px", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

print("Areas in pixels (component_id, area_px):")
for i, area in areas:
    print(i, area)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Mask"); plt.axis("off"); plt.imshow(mask, cmap="gray")
plt.subplot(1,2,2); plt.title("Labeled / BBoxes"); plt.axis("off"); plt.imshow(vis)
plt.tight_layout(); plt.show()

cv2.imwrite("q10c_labeled.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
