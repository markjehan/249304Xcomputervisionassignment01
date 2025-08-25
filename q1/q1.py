import matplotlib.pyplot as plt
from PIL import Image

control_pts = [
    (0, 0),
    (50, 50),
    (50, 100),
    (150, 255),
    (150, 150),
    (255, 255),
]

def build_lut_from_control_points(points):
    """
    Build a 256-length LUT from piecewise-linear control points that may
    include vertical steps (duplicate x with different y). For a vertical
    step at x = a (i.e., ... (a,y_low), (a,y_high) ...), we map r==a to
    the 'later' point's y (the second tuple), i.e., y_high in this list.
    """
    lut = [0]*256

    segs = []
    for i in range(len(points)-1):
        x0, y0 = points[i]
        x1, y1 = points[i+1]
        segs.append((x0, y0, x1, y1))

    for r in range(256):
        y = None

        for (x0, y0, x1, y1) in segs:
            if x0 == x1:
                if r == x0:
                    y = y1
                    break
                continue

            left, right = (x0, x1) if x0 < x1 else (x1, x0)
            y_left, y_right = (y0, y1) if x0 < x1 else (y1, y0)

            if left <= r <= right:
                t = (r - left) / float(right - left)
                y = int(round(y_left + t * (y_right - y_left)))
                break

        if y is None:
            if r < points[0][0]:
                y = points[0][1]
            else:
                y = points[-1][1]

        lut[r] = max(0, min(255, y))

    return lut

def main():
    try:
        img = Image.open("emma.jpg").convert("L")
    except Exception as e:
        print("Could not open emma.jpg:", e)
        return

    lut = build_lut_from_control_points(control_pts)

    w, h = img.size
    inp = img.load()
    out_img = Image.new("L", (w, h))
    outp = out_img.load()
    for y in range(h):
        for x in range(w):
            outp[x, y] = lut[inp[x, y]]

    out_img.save("emma_piecewise.png")
    print("Saved: emma_piecewise.png")

    xs, ys = zip(*control_pts)

    plt.figure(figsize=(6,6))
    plt.plot(range(256), lut, linewidth=2, label="LUT curve")
    plt.plot(xs, ys, "o--", label="control points")
    plt.title("Piecewise Intensity Transformation (with jumps)")
    plt.xlabel("Input Intensity (r)")
    plt.ylabel("Output Intensity (s)")
    plt.grid(True)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original (emma.jpg)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(out_img, cmap="gray", vmin=0, vmax=255)
    plt.title("Transformed")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
