import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG = "einstein.png"

def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    k = kernel.astype(np.float32)               
    H, W = img.shape
    p = np.pad(img, 1, mode="constant")         
    out = np.empty((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            out[y, x] = (p[y:y+3, x:x+3] * k).sum()
    return out

def norm_u8(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    m = a.max()
    if m > 0:
        a = a * (255.0 / m)
    return a.astype(np.uint8)


img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(IMG)

Kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], np.float32)
Ky = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], np.float32)

gx = conv2d(img, Kx)
gy = conv2d(img, Ky)

mag = np.hypot(gx, gy)

gx_u8, gy_u8, mag_u8 = map(norm_u8, (gx, gy, mag))
cv2.imwrite("sobel_b_gx.png", gx_u8)
cv2.imwrite("sobel_b_gy.png", gy_u8)
cv2.imwrite("sobel_b_mag.png", mag_u8)


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(gx_u8, cmap="gray"); plt.title("Gx (manual)"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gy_u8, cmap="gray"); plt.title("Gy (manual)"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(mag_u8, cmap="gray"); plt.title("Magnitude"); plt.axis("off")
plt.tight_layout(); plt.show()
