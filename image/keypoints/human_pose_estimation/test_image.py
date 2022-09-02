import sys, os
from tensorflow import keras
import numpy as np
import simplejson as json
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt

from data_loader import Dataloader
from util import visualize_keypoints

if len(sys.argv) != 3:
    print(f"Usage {sys.argv[0]} <model> <img-file>")
    exit()

model_dir = sys.argv[1]
image_file = sys.argv[2]

print(f"Loading model... {model_dir}")
model = keras.models.load_model(model_dir)

hyper_param_file = os.path.join(model_dir, "params.json")
params = {}
with open(hyper_param_file, "r") as f:
    params = json.load(f)

print(f"Got hyperparams {params}")

num_keypoints = params["num_keypoints"] if "num_keypoints" in params else 13

img_size = params["image_size"]
test_aug = iaa.Sequential([iaa.Resize(img_size, interpolation="linear")])
dataloader = Dataloader(
    image_dir=None, num_keypoints=num_keypoints, aug=test_aug, image_size=img_size
)

frame = dataloader.read_image_from_disk(image_file)
(frame) = test_aug(image=frame)
frame = frame.reshape(1, img_size, img_size, 3)
print(f"frame {frame.shape}")


outputs = model.predict(frame)
frame = frame.reshape(img_size, img_size, 3)
outputs = outputs.reshape((img_size, img_size, dataloader.num_outputs()))
print(outputs.shape)

kps = []
for i in range(dataloader.num_outputs()):
    heatmap = np.array(outputs[:, :, i])
    # print("heatmap.shape", heatmap.shape)
    # print("heatmap", heatmap)

    # np.save(f"heatmap_{i}.npy", heatmap)

    x = np.argmax(np.amax(heatmap, axis=1))
    y = np.argmax(np.amax(heatmap, axis=0))
    val = heatmap[x][y]
    format_val = "{:.2f}".format(val)

    if val > 0.5:
        print(f"Got kp[{i}] {x},{y} val {format_val}")

        kps.append(x)
        kps.append(y)
        kps.append(1)
        plt.imshow(heatmap, interpolation="nearest")
        plt.show()
    else:
        kps.append(x)
        kps.append(y)
        kps.append(0)
        # kps.append(1)

        # heatmap -= heatmap.mean()
        # heatmap /= heatmap.std()
        # heatmap *=  64
        # heatmap += 128

        # x = np.argmax(np.amax(heatmap, axis=1))
        # y = np.argmax(np.amax(heatmap, axis=0))
        # val = heatmap[x][y]
        # format_val = "{:.2f}".format(val)

        # print(f"Got suboptimal kp[{i}] {x},{y} val {format_val}")

        # plt.imshow(heatmap, interpolation='nearest')
        # plt.show()

    # break

# visualize_keypoints([frame], [kps])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
[ax.axis("off") for ax in np.ravel(axes)]

print(axes)
ax_orig = axes[0]
ax_all = axes[1]

ax_orig.imshow(frame)
ax_all.imshow(frame)

n_keypoints = dataloader.num_outputs()
step = 3
current_keypoint = np.array(kps)
# Since the last entry is the visibility flag, we discard it.
for i in range(0, n_keypoints * step, step):
    x = float(current_keypoint[i])
    y = float(current_keypoint[i + 1])
    is_visible = current_keypoint[i + 2]
    if is_visible:
        ax_all.scatter([x], [y], c="#FF0000", marker="x", s=50, linewidths=5)

plt.tight_layout(pad=2.0)
plt.show()
