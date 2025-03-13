from convert_aloha_data_to_lerobot_pi0 import load_raw_images_per_camera
from pathlib import Path
import tqdm, h5py
import numpy as np

def main():
    raw_dir = Path("~/data/aloha_data_raw/aloha_mobile_right_single_rigid").expanduser()
    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    # episodes = range(len(hdf5_files))
    # for ep_idx in tqdm.tqdm(episodes):
    ep_path = hdf5_files[10]

    with h5py.File(ep_path, "r") as ep:
        for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for i, data in enumerate(ep[f"/observations/images/{camera}"]):
                img = cv2.imdecode(data, 1)
                cropped_img = crop_image(img)
                # cropped_imgs_array.append(cropped_img)
                print(f"After cropping, the {i}th image has shape of {cropped_img.shape}.")
            # cropped_imgs_array = np.array(cropped_imgs_array)


def crop_image(image, desired_width=640):
    h, w, _ = image.shape
    margin = (w - desired_width) // 2
    cropped_image = image[:, margin:margin+desired_width]
    return cropped_image


if __name__ == "__main__":
    main()