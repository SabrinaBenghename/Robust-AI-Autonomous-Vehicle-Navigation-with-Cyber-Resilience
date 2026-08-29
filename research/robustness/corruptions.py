from PIL import Image, ImageFilter, ImageEnhance
import torch
from torchvision import transforms


# ============================================================
# REPRODUCIBLE ROBUSTNESS CONDITIONS
# ============================================================

IMAGE_TO_TENSOR = transforms.ToTensor()
TENSOR_TO_IMAGE = transforms.ToPILImage()


# ------------------------------------------------------------
# NOISE
# ------------------------------------------------------------

def add_noise(image, noise_std=0.05):

    tensor = IMAGE_TO_TENSOR(image)

    noise = torch.randn_like(tensor) * noise_std

    tensor = tensor + noise

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0
    )

    return TENSOR_TO_IMAGE(tensor)


# ------------------------------------------------------------
# BLUR
# ------------------------------------------------------------

def add_blur(image, radius=2):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


# ------------------------------------------------------------
# LOW CONTRAST
# ------------------------------------------------------------

def reduce_contrast(image, factor=0.5):

    enhancer = ImageEnhance.Contrast(
        image
    )

    return enhancer.enhance(
        factor
    )


# ------------------------------------------------------------
# OCCLUSION
# ------------------------------------------------------------

def add_occlusion(
    image,
    left_ratio=0.40,
    right_ratio=0.60,
    top_ratio=0.45,
    bottom_ratio=0.65
):

    image = image.copy()

    width, height = image.size

    left = int(
        width * left_ratio
    )

    right = int(
        width * right_ratio
    )

    top = int(
        height * top_ratio
    )

    bottom = int(
        height * bottom_ratio
    )

    pixels = image.load()

    for x in range(left, right):

        for y in range(top, bottom):

            pixels[x, y] = (
                0,
                0,
                0
            )

    return image


# ------------------------------------------------------------
# APPLY CONDITION
# ------------------------------------------------------------

def apply_condition(
    image,
    condition
):

    if condition == "clean":

        return image

    elif condition == "noise":

        return add_noise(
            image,
            noise_std=0.05
        )

    elif condition == "blur":

        return add_blur(
            image,
            radius=2
        )

    elif condition == "low_contrast":

        return reduce_contrast(
            image,
            factor=0.5
        )

    elif condition == "occlusion":

        return add_occlusion(
            image
        )

    else:

        raise ValueError(
            f"Unknown robustness condition: {condition}"
        )