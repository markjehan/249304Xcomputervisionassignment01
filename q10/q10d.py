import cv2, numpy as np

IMG = "sapphire.jpg"

F_MM = 8.0        
Z_MM = 480.0      
PIXEL_SIZE_MM = 0.004  

mm_per_px = (Z_MM / F_MM) * PIXEL_SIZE_MM
AREA_PER_PX_MM2 = mm_per_px ** 2

bgr = cv2.imread(IMG)
if bgr is None: raise FileNotFoundError(IMG)

hsv = cv2.cvtColor(cv2.GaussianBlur(bgr,(5,5),0), cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([90,40,30]), np.array([140,255,255]))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=1)
H, W = mask.shape
ff = mask.copy(); border = np.zeros((H+2,W+2), np.uint8)
cv2.floodFill(ff, border, (0,0), 255); holes = cv2.bitwise_not(ff) & mask
mask = cv2.bitwise_or(mask, holes)

count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
areas_px = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 100]

areas_mm2 = [(i, a_px * AREA_PER_PX_MM2) for i, a_px in areas_px]

print(f"Assumptions: f={F_MM} mm, Z={Z_MM} mm, pixel_size={PIXEL_SIZE_MM*1000:.1f} µm")
print(f"Scale: {mm_per_px:.6f} mm/px   ->   {AREA_PER_PX_MM2:.6f} mm^2/px\n")

for i, a_px in areas_px:
    a_mm2 = a_px * AREA_PER_PX_MM2
    print(f"Component {i}:  area = {a_px} px  ≈  {a_mm2:.2f} mm²")

with open("q10d_areas_mm2.csv", "w", encoding="utf-8") as f:
    f.write("component_id,area_px,area_mm2\n")
    for (i, a_px), (_, a_mm2) in zip(areas_px, areas_mm2):
        f.write(f"{i},{a_px},{a_mm2:.6f}\n")
print("\nSaved: q10d_areas_mm2.csv")
