import cv2
import matplotlib.pyplot as plt

INPUT_IMAGE = "jeniffer.jpg"
OUTPUT_MASK = "jeniffer_mask.png"
PLANE = "V"   

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not read", INPUT_IMAGE)
        return
    print("Image loaded. Shape:", bgr.shape)

    print("Converting BGR -> HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    if PLANE.upper() == "S":
        plane = S
        print("Using S (Saturation) channel for mask.")
    else:
        plane = V
        print("Using V (Value) channel for mask.")

    print("Applying Otsu threshold...")
    thresh_val, mask = cv2.threshold(
        plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("Otsu threshold value chosen automatically:", thresh_val)

    ok = cv2.imwrite(OUTPUT_MASK, mask)
    if ok:
        print("Saved binary mask to:", OUTPUT_MASK)
    else:
        print("Warning: could not save mask image.")

    print("Showing original plane and mask...")
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(plane, cmap="gray")
    plt.title(PLANE + " plane")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Foreground mask (binary)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
