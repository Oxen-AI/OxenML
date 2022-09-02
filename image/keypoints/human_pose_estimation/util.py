from matplotlib import pyplot as plt
from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np


def visualize_keypoints(images, keypoints):
    print(len(images))
    print(len(keypoints))
    fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(16, 12))
    [ax.axis("off") for ax in np.ravel(axes)]

    for (ax_orig, ax_all), image, current_keypoint in zip(axes, images, keypoints):
        ax_orig.imshow(image)
        ax_all.imshow(image)

        # If the keypoints were formed by `imgaug` then the coordinates need
        # to be iterated differently.
        if isinstance(current_keypoint, KeypointsOnImage):
            for idx, kp in enumerate(current_keypoint.keypoints):
                ax_all.scatter(
                    [kp.x], [kp.y], c="#FFFFFF", marker="x", s=50, linewidths=5
                )
        elif isinstance(current_keypoint, np.array):
            print("GOT NP ARRAY")
            print(current_keypoint)
        else:
            n_keypoints = 17
            step = 3
            current_keypoint = np.array(current_keypoint)
            # Since the last entry is the visibility flag, we discard it.
            for i in range(0, n_keypoints * step, step):
                x = float(current_keypoint[i])
                y = float(current_keypoint[i + 1])
                ax_all.scatter([x], [y], c="#FFFFFF", marker="x", s=50, linewidths=5)

    plt.tight_layout(pad=2.0)
    plt.show()
