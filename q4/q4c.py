import cv2
import matplotlib.pyplot as plt
import math

INPUT_IMAGE = "spider.png"
SIGMA = 70.0
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]  
CHOSEN_ALPHA = 0.8  

def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def vibrance_pixel(s_value, alpha, sigma):
    diff = (s_value - 128.0)
    exp_term = math.exp(-(diff * diff) / (2.0 * sigma * sigma))
    bump = alpha * 128.0 * exp_term
    new_val = s_value + bump
    return clamp_0_255(new_val)

def vibrance_S_channel(S, alpha, sigma):
    h = S.shape[0]
    w = S.shape[1]
    out = S.copy()
    y = 0
    while y < h:
        x = 0
        while x < w:
            val = int(S[y, x])
            out[y, x] = vibrance_pixel(val, alpha, sigma)
            x = x + 1
        y = y + 1
    return out

def main():
    print("Opening image:", INPUT_IMAGE)
    bgr = cv2.imread(INPUT_IMAGE)
    if bgr is None:
        print("Error: could not read", INPUT_IMAGE)
        return
    print("Image opened. Shape:", bgr.shape)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    cols = len(ALPHAS) + 1
    plt.figure(figsize=(3.5 * cols, 7))

    plt.subplot(2, cols, 1)
    plt.imshow(rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, cols, cols + 1)
    plt.imshow(S, cmap="gray", vmin=0, vmax=255)
    plt.title("S (orig)")
    plt.axis("off")

    index = 2
    for a in ALPHAS:
        print("Applying vibrance with alpha =", a)
        S_v = vibrance_S_channel(S, a, SIGMA)
        hsv_new = cv2.merge([H, S_v, V])
        bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        rgb_new = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2RGB)

        plt.subplot(2, cols, index)
        plt.imshow(rgb_new)
        plt.title("alpha=" + str(a))
        plt.axis("off")

        plt.subplot(2, cols, cols + index)
        plt.imshow(S_v, cmap="gray", vmin=0, vmax=255)
        plt.title("S (alpha=" + str(a) + ")")
        plt.axis("off")

        index = index + 1

    plt.tight_layout()
    plt.show()

    print("Chosen alpha (report this):", CHOSEN_ALPHA)

if __name__ == "__main__":
    main()
