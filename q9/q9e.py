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

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
areas = stats[1:, cv2.CC_STAT_AREA]  
area_thresh = max(50, int(np.percentile(areas, 5))) if areas.size else 50  

mask_big = np.zeros_like(cleaned)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
        mask_big[labels == i] = 255

count, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(mask_big, connectivity=8)
n_grains = count - 1  

h, w = labels2.shape
label_img = np.zeros((h, w, 3), np.uint8)
rng = np.random.default_rng(123)
palette = rng.integers(0, 255, size=(count, 3), dtype=np.uint8)
for i in range(1, count):
    label_img[labels2 == i] = palette[i]

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("Cleaned Binary"); plt.axis("off"); plt.imshow(mask_big, cmap="gray")
plt.subplot(1,2,2); plt.title(f"Labeled Components (n={n_grains})"); plt.axis("off"); plt.imshow(label_img)
plt.tight_layout(); plt.show()

cv2.imwrite("q9e_labeled.png", cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR))
print(f"Estimated number of rice grains: {n_grains}")
