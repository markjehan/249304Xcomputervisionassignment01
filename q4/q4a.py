import cv2
import matplotlib.pyplot as plt

INPUT_IMAGE = "spider.png"
H_OUT = "spider_H.png"
S_OUT = "spider_S.png"
V_OUT = "spider_V.png"

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if bgr is None:
        print("Error: could not read", INPUT_IMAGE, ". Check the path/name.")
        return
    print("Image opened. Shape (H, W, C):", bgr.shape)

    print("Converting to RGB for display...")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Converting to HSV...")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    print("Splitting channels...")
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    print("H channel range (expected 0..179): min =", int(H.min()), "max =", int(H.max()))
    print("S channel range (expected 0..255): min =", int(S.min()), "max =", int(S.max()))
    print("V channel range (expected 0..255): min =", int(V.min()), "max =", int(V.max()))

    print("Saving channels...")
    ok_h = cv2.imwrite(H_OUT, H)
    ok_s = cv2.imwrite(S_OUT, S)
    ok_v = cv2.imwrite(V_OUT, V)
    if ok_h: print("Saved:", H_OUT)
    else:    print("Warning: could not save", H_OUT)
    if ok_s: print("Saved:", S_OUT)
    else:    print("Warning: could not save", S_OUT)
    if ok_v: print("Saved:", V_OUT)
    else:    print("Warning: could not save", V_OUT)

    print("Showing 2x2 view...")
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(rgb)
    plt.title("Original (RGB)")
    plt.axis("off")

    
    plt.subplot(2, 2, 2)
    plt.imshow(H, cmap="hsv")
    plt.title("Hue (H)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(S, cmap="gray", vmin=0, vmax=255)
    plt.title("Saturation (S)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(V, cmap="gray", vmin=0, vmax=255)
    plt.title("Value (V)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
