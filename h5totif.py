from pathlib import Path

import tifffile
import h5py
import numpy as np


def h5_to_tif(h5_file: Path | str, output_tif: Path | str) -> bool:
    """
    Convert a dataset from an HDF5 file to a TIFF file.

    Parameters:
    h5_file (str): Path to the input HDF5 file.
    output_tif (str): Path to the output TIFF file.
    """
    # Open the HDF5 file
    with h5py.File(h5_file, 'r') as hf:
        # Read the dataset
        det = hf["run/scan00001/det"]
        det_key = list(det.keys())[0]

        img: np.ndarray = det[det_key]["data"][()].squeeze().astype(np.float32)
        if img.ndim != 2:
            print(f"Dataset from {h5_file} is not 2D: shape={img.shape}")
            return False

        # Save the data as a TIFF file
        tifffile.imwrite(output_tif, img, imagej=True)
        return True

if __name__ == "__main__":
    h5_root: Path = Path(r"C:\WorkSpace\05_Resources\Data\DFXM\ue_260617_DFXM\dat")
    tif_root: Path = Path(r"C:\WorkSpace\05_Resources\Data\DFXM\ue_260617_DFXM\analysis\tif_files")
    tif_root.mkdir(parents=True, exist_ok=True)
    for file in h5_root.rglob(pattern="**/*.h5"):
        output_tif: Path = tif_root / f"{file.parent.name}_{file.stem}.tif"
        print(f"\n{file} -> \n{output_tif}")
        if not h5_to_tif(file, output_tif):
            print(f"Failed to convert {file}")