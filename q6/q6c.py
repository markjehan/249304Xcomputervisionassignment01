import cv2
import numpy as np
import matplotlib.pyplot as plt  

IMG = "einstein.png"

def norm_u8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a *= 255.0 / m
    return a.astype(np.uint8)

img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Couldn't open {IMG}. Put it next to this script.")

s = np.array([1, 2, 1],  dtype=np.float32)   
d = np.array([1, 0, -1], dtype=np.float32)   
print("Sobel-x kernel via outer(s, d):\n", np.outer(s, d))

gx = cv2.sepFilter2D(img, ddepth=cv2.CV_32F, kernelX=d, kernelY=s)  
gy = cv2.sepFilter2D(img, ddepth=cv2.CV_32F, kernelX=s, kernelY=d)  
mag = np.hypot(gx, gy)

gx_u8, gy_u8, mag_u8 = map(norm_u8, (gx, gy, mag))
cv2.imwrite("q6c_gx.png", gx_u8)
cv2.imwrite("q6c_gy.png", gy_u8)
cv2.imwrite("q6c_mag.png", mag_u8)

tiled = np.hstack([img, gx_u8, gy_u8, mag_u8])
cv2.namedWindow("Original | Gx | Gy | Magnitude", cv2.WINDOW_NORMAL)
cv2.imshow("Original | Gx | Gy | Magnitude", tiled)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL); cv2.imshow("Original", img)
cv2.namedWindow("Gx (separable)", cv2.WINDOW_NORMAL); cv2.imshow("Gx (separable)", gx_u8)
cv2.namedWindow("Gy (separable)", cv2.WINDOW_NORMAL); cv2.imshow("Gy (separable)", gy_u8)
cv2.namedWindow("Magnitude", cv2.WINDOW_NORMAL); cv2.imshow("Magnitude", mag_u8)

print("Press any key in an image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
