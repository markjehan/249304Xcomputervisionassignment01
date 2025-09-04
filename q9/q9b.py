import cv2
import matplotlib.pyplot as plt

IMG = "rice.png"

g = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
if g is None:
    raise FileNotFoundError(IMG)

gauss = cv2.GaussianBlur(g, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)

nlm = cv2.fastNlMeansDenoising(
    g, None,      
    h=10,          
    templateWindowSize=7,
    searchWindowSize=21
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.title("Original"); plt.axis("off"); plt.imshow(g, cmap="gray")
plt.subplot(1, 3, 2); plt.title("Gaussian σ=1.2"); plt.axis("off"); plt.imshow(gauss, cmap="gray")
plt.subplot(1, 3, 3); plt.title("Non-Local Means (final)"); plt.axis("off"); plt.imshow(nlm, cmap="gray")
plt.tight_layout(); plt.show()

cv2.imwrite("q9b_preprocessed.png", nlm)
