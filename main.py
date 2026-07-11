from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def main():
    root: Path = Path(r"C:\WorkSpace\05_Resources\Data\DFXM\ue_260617_DFXM\dat")
    file1: Path = root / "20260618/260618_shockwave_01_00001.h5"
    file2: Path = root / "20260618/260618_shockwave_01_00002.h5"

    with h5py.File(file1, "r") as hf1, h5py.File(file2, "r") as hf2:
        img1: np.ndarray = hf1["run/scan00001/det/eh1hama_img/data"][:].squeeze()
        img2: np.ndarray = hf2["run/scan00001/det/eh1hama_img/data"][:].squeeze()
        print("image shape:", img1.shape)
        print("image shape:", img2.shape)
    bg_zone_off: np.ndarray = img1[0:50, 0:50]
    bg_zone_on: np.ndarray = img2[0:50, 0:50]
    mean_bg_off: float = np.mean(bg_zone_off)
    mean_bg_on: float = np.mean(bg_zone_on)

    scale_factor: float = mean_bg_on / mean_bg_off
    img1_scaled: np.ndarray = img1 * scale_factor
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    ax[0, 0].imshow(np.log1p(img1))
    ax[0, 1].imshow(np.log1p(img2))
    ax[1, 0].imshow(np.log1p(img2 / img1))
    ax[1, 1].imshow(np.log1p(img2 - img1_scaled))
    plt.show()


if __name__ == "__main__":
    main()
