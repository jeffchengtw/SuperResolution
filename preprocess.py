import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert ``PIL.Image`` to Tensor.
    Args:
        image (np.ndarray): The image data read by ``PIL.Image``
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.
    Returns:
        Normalized image data
    Examples:
        >>> image = Image.open("image.bmp")
        >>> tensor_image = image2tensor(image, range_norm=False, half=False)
    """
    tensor = F.to_tensor(image)

    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()

    return tensor

def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Converts ``torch.Tensor`` to ``PIL.Image``.
    Args:
        tensor (torch.Tensor): The image that needs to be converted to ``PIL.Image``
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.
    Returns:
        Convert image data to support PIL library
    Examples:
        >>> tensor = torch.randn([1, 3, 128, 128])
        >>> image = tensor2image(tensor, range_norm=False, half=False)
    """
    if range_norm:
        tensor = tensor.add_(1.0).div_(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")

    return image
    
def convert_rgb_to_y(image: Any) -> Any:
    """Convert RGB image or tensor image data to YCbCr(Y) format.
    Args:
        image: RGB image data read by ``PIL.Image''.
    Returns:
        Y image array data.
    """
    if type(image) == np.ndarray:
        return 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
    elif type(image) == torch.Tensor:
        if len(image.shape) == 4:
            image = image.squeeze_(0)
        return 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
    else:
        raise Exception("Unknown Type", type(image))