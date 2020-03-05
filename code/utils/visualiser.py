import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils
import torch

def get_checkerboard(size):
    a = np.arange(size)
    x, y = np.meshgrid(a, a)
    grid = np.stack([x, y])

    x = (grid[0] / 16).astype(np.uint8) % 2
    y = (grid[1] / 16).astype(np.uint8) % 2
    checker = np.logical_xor(x, y)[..., np.newaxis]

    color1 = checker * np.array([0.6, 0.6, 0.6]).reshape(1, 1, 3)
    color2 = np.invert(checker) * np.array([0.7, 0.7, 0.7]).reshape(1, 1, 3)

    color = color1 + color2
    return color

def slice_volume(voxels, slice_idx, is_batch=True, flip=True):
    slice = voxels[:, :, slice_idx, :, :]
    show_image(slice, is_batch=is_batch, flip=flip)


def show_image(image, is_batch=True, flip=True, resolution=-1, path=None):
    if resolution != -1:
        image = torch.nn.functional.interpolate(image, size=(resolution, resolution), mode='nearest')

    if type(image).__module__ == 'torch':
        image = utils.pytorch_to_numpy(image, is_batch, flip)
    bs = image.shape[0]
    size = image.shape[1]
    channel = image.shape[-1]

    if channel == 2:
        image = np.concatenate((image, np.zeros_like(image[...,0:1])), axis=-1)

    if channel == 4:
        # add checkerboard
        checker = get_checkerboard(size)
        alpha = image[..., [3]]
        rgb = image[..., 0:3]
        image = checker * (1 - alpha) + rgb * alpha

    cmap = None

    if image.ndim == 3:
        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(image), cmap=cmap)
        plt.colorbar(im, ax=ax)

    elif bs == 1:
        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(image[0]), cmap=cmap)
        # plt.colorbar(im, ax=ax)
    else:
        fig, axes = plt.subplots(1, image.shape[0], figsize=(150, 150))
        for idx, axis in enumerate(axes):
            im = axis.imshow(np.squeeze(image[idx]), cmap=cmap)
            # plt.colorbar(im, ax=axis)


    if path is not None:
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight',transparent=True, pad_inches=0)
    else:
        plt.show()
