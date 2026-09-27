import cv2
import torch
import numpy as np


def apply_postprocess(tensor: torch.Tensor) -> torch.Tensor:
    image = (tensor[0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    var_d = 7
    var_r = 125
    image_filtered = cv2.bilateralFilter(image, d=var_d, sigmaColor=var_r, sigmaSpace=var_r)

    mask_size = 15
    coef = 0.9
    C = int((1 - coef) * 255)

    thresholded = cv2.adaptiveThreshold(
        image_filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=mask_size,
        C=C
    )

    return torch.from_numpy(thresholded.astype(np.float32) / 255.0).unsqueeze(0)
