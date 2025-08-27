import matplotlib.pyplot as plt
from PIL import Image

def clamp_0_255(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)

def make_int(v):
    try:
        return int(v)
    except:
        return 0

def clamp_point(x, y):
    x = make_int(x)
    y = make_int(y)
    if x < 0: x = 0
    if x > 255: x = 255
    if y < 0: y = 0
    if y > 255: y = 255
    return (x, y)

def ensure_endpoints(points):
    if len(points) == 0:
        points = [(0, 0), (255, 255)]
        return points

    first_x, first_y = points[0]
    last_x, last_y = points[-1]

    if first_x != 0:
        points = [(0, first_y)] + points
    if last_x != 255:
        points = points + [(255, last_y)]
    return points

def build_lut_from_points_beginner(control_pts):
    """
    Beginner-style LUT builder from control points.
    - Uses simple loops and lists.
    - Respects vertical jumps when two points share the same x.
    - Fills missing values by simple forward/backward fill.
    Returns:
        lut_list: list of 256 ints
        r_vals:   [0..255]
        s_vals:   same as lut_list (for plotting)
    """
    print("Building LUT from control points (beginner style)...")

    cps = []
    i = 0
    while i < len(control_pts):
        x, y = control_pts[i]
        cps.append(clamp_point(x, y))
        i = i + 1

    cps = ensure_endpoints(cps)

    lut = [None] * 256

    j = 0
    while j < len(cps) - 1:
        x0, y0 = cps[j]
        x1, y1 = cps[j + 1]

        if x1 == x0:
            lut[x0] = clamp_0_255(y1)
        else:
            step = 1
            if x1 < x0:
                step = -1

            dx = (x1 - x0)
            if dx < 0:
                dx = -dx

            x = x0
            while True:
                denom = (cps[j + 1][0] - cps[j][0])  
                if denom == 0:
                    t = 0.0
                else:
                    t = (x - x0) / float(denom)

                y = y0 + (y1 - y0) * t
                lut[x] = clamp_0_255(y)

                if x == x1:
                    break
                x = x + step

        j = j + 1

    last_seen = None
    i = 0
    while i < 256:
        if lut[i] is None:
            if last_seen is not None:
                lut[i] = last_seen
        else:
            last_seen = lut[i]
        i = i + 1

    next_seen = None
    i = 255
    while i >= 0:
        if lut[i] is None:
            if next_seen is not None:
                lut[i] = next_seen
        else:
            next_seen = lut[i]
        i = i - 1

    i = 0
    while i < 256:
        lut[i] = clamp_0_255(lut[i] if lut[i] is not None else 0)
        i = i + 1

    r_vals = list(range(256))
    s_vals = list(lut)
    print("LUT ready with", len(lut), "entries.")
    return lut, r_vals, s_vals

def apply_lut_pixel_by_pixel(gray_img, lut):
    """
    Beginner-style pixel-by-pixel LUT application using PIL .load().
    gray_img: PIL Image in mode "L"
    """
    print("Applying LUT to image (pixel-by-pixel, may be slow)...")
    w, h = gray_img.size
    out_img = Image.new("L", (w, h))
    in_px = gray_img.load()
    out_px = out_img.load()

    y = 0
    while y < h:
        x = 0
        while x < w:
            v = in_px[x, y]  # 0..255
            out_px[x, y] = lut[v]
            x = x + 1
        y = y + 1

    return out_img

control_pts_wm = [
(0, 0), 
(110, 100), 
(180, 170), 
(200, 255),
(255,255)

]

control_pts_gm = [
    (0, 0),
    (80, 60),
    (120, 180),  
    (180, 205),
    (255, 255)
]

def main():
    INPUT_IMAGE = "brain_proton_density_slice.png"
    WM_OUT = "wm_from_ctrlpts.png"
    GM_OUT = "gm_from_ctrlpts.png"

    print("Opening input image:", INPUT_IMAGE)
    try:
        img = Image.open(INPUT_IMAGE).convert("L")
    except Exception as e:
        print("Failed to open image. Error:", e)
        return

    print("Image size:", img.size)

    print("Building White Matter LUT...")
    lut_wm, r_wm, s_wm = build_lut_from_points_beginner(control_pts_wm)

    print("Building Gray Matter LUT...")
    lut_gm, r_gm, s_gm = build_lut_from_points_beginner(control_pts_gm)

    out_img_wm = apply_lut_pixel_by_pixel(img, lut_wm)
    out_img_gm = apply_lut_pixel_by_pixel(img, lut_gm)

    out_img_wm.save(WM_OUT)
    out_img_gm.save(GM_OUT)
    print("Saved:", WM_OUT)
    print("Saved:", GM_OUT)

    print("Plotting transform curves...")
    plt.figure(figsize=(7, 7))
    plt.plot(r_wm, s_wm, label="White Matter", linewidth=2)
    plt.plot(r_gm, s_gm, label="Gray Matter", linewidth=2, linestyle="--")

    wmx = [p[0] for p in control_pts_wm]
    wmy = [p[1] for p in control_pts_wm]
    gmx = [p[0] for p in control_pts_gm]
    gmy = [p[1] for p in control_pts_gm]
    plt.scatter(wmx, wmy, s=45)
    plt.scatter(gmx, gmy, s=45)

    plt.title("Intensity Transform (Control-Point Based)")
    plt.xlabel("Input intensity (r)")
    plt.ylabel("Output intensity (s)")
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Showing images...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(out_img_wm, cmap="gray", vmin=0, vmax=255)
    plt.title("White Matter")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(out_img_gm, cmap="gray", vmin=0, vmax=255)
    plt.title("Gray Matter")
    plt.axis("off")

    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
