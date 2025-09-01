import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def ssd(a, b) -> float:
    """Sum of squared differences using OpenCV's L2SQR norm."""
    return float(cv2.norm(a, b, cv2.NORM_L2SQR))

def compare_pair(low_path: str, high_path: str) -> None:
    lo = cv2.imread(low_path)
    hi = cv2.imread(high_path)

    if lo is None or hi is None:
        print(f"Skipping (missing): {low_path} or {high_path}")
        return

    H, W = hi.shape[:2]
    nn   = cv2.resize(lo, (W, H), interpolation=cv2.INTER_NEAREST)
    lin  = cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)

    d_nn  = ssd(hi, nn)
    d_lin = ssd(hi, lin)

    print(f"{low_path} -> {high_path}")
    print(f"  SSD nearest : {d_nn:.2f}")
    print(f"  SSD bilinear: {d_lin:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(hi, cv2.COLOR_BGR2RGB));  ax[0].set_title("Original");            ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(nn, cv2.COLOR_BGR2RGB));  ax[1].set_title("Upscaled (Nearest)");  ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(lin, cv2.COLOR_BGR2RGB)); ax[2].set_title("Upscaled (Bilinear)"); ax[2].axis("off")
    fig.suptitle(Path(high_path).name)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
  
    PAIRS = [
        ("im01small.png", "im01.png"),
        ("im02small.png", "im02.png"),
        ("im03small.png", "im03.png"),
    ]
    for lo_fp, hi_fp in PAIRS:
        compare_pair(lo_fp, hi_fp)
