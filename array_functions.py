import numpy as np
import cv2
from scipy.ndimage import center_of_mass


def compress_array(arr):
    res_arr = np.zeros((28, 28))

    for i in range(0, len(arr), 10):
        for j in range(0, len(arr[0]), 10):
            s = 0

            for k in range(i, i + 10):
                for h in range(j, j + 10):
                    s += arr[k][h]

            s /= 100
            s *= 255
            res_arr[i // 10][j // 10] = int(s)

    return res_arr


def stretch_array(arr):
    res_arr = np.zeros((280, 280))

    for i in range(280):
        for j in range(280):
            res_arr[i][j] = arr[i // 10][j // 10]

    return res_arr


def crop_to_bbox(bw, margin=3):
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return bw, (0, 0, bw.shape[1] - 1, bw.shape[0] - 1)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(bw.shape[1] - 1, x1 + margin)
    y1 = min(bw.shape[0] - 1, y1 + margin)
    return bw[y0:y1 + 1, x0:x1 + 1], (x0, y0, x1, y1)


def resize_keep_aspect_and_pad(img, target=28, inner_margin=2):
    h, w = img.shape
    max_side = target - 2 * inner_margin

    if max(h, w) == 0:
        return np.full((target, target), 0, dtype=img.dtype)
    scale = max_side / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    if nh == 0: nh = 1
    if nw == 0: nw = 1
    resized = cv2.resize(img, (nw, nh))

    top = (target - nh) // 2
    bottom = target - nh - top
    left = (target - nw) // 2
    right = target - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def center_by_centroid(img):
    bw = (img > 0).astype(np.uint8)
    if bw.sum() == 0:
        return img
    cy, cx = center_of_mass(bw)
    if np.isnan(cx) or np.isnan(cy):
        return img
    h, w = img.shape
    shift_x = int(round(w / 2 - cx))
    shift_y = int(round(h / 2 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    out = cv2.warpAffine(img, M, (w, h), borderValue=0)
    return out


def normalize_to_mnist(img_canvas, target=28, margin=6, inner_margin=2):
    """
    img_canvas: HxW uint8 grayscale (0..255)
    returns: 28x28 float32 (0..1) ready for model
    """

    cropped, _ = crop_to_bbox(img_canvas, margin=margin)
    resized_padded = resize_keep_aspect_and_pad(cropped, target=target, inner_margin=inner_margin)
    centered = center_by_centroid(resized_padded)
    final = cv2.GaussianBlur(centered, (3, 3), 0)
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    return final
