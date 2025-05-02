import cv2
import numpy as np
import aiccm
import torch

from . import device

from typing import Literal

COLOR = Literal['gray', 'color']
MASK_MODE = Literal['green', 'white', 'black']
range_1 = np.array([0, 200, 0])
range_2 = np.array([50, 255, 50])


def load_image(path, color: COLOR = 'gray'):
    """ 加载一张图片 """
    with open(path, 'rb') as file:
        file_data = file.read()
        image_array = np.frombuffer(file_data, np.uint8)
        match color:
            case 'color':
                flags = cv2.IMREAD_COLOR
            case 'gray':
                flags = cv2.IMREAD_GRAYSCALE
            case _:
                flags = cv2.IMREAD_UNCHANGED
        image_raw = cv2.imdecode(image_array, flags)

    return image_raw


def parse_input(ori_path, mask_path, mask_mode: MASK_MODE = 'green'):
    """
    解析输入
    :param ori_path: 待修复图像路径
    :param mask_path: 待修复区域掩码图像路径
    :param mask_mode: 掩码模式
    :return:
    """

    ori = load_image(ori_path)
    ori_img = ori.copy()
    ori = ori.astype(np.float32) / 127.5 - 1

    match mask_mode:
        case 'green':
            mask = load_image(mask_path, 'color')
            mask = cv2.inRange(mask, range_1, range_2)
            mask = mask.astype('uint8')
            mask = 255 - mask
        case 'white':
            mask = load_image(mask_path, 'gray')
            mask = 255 - mask
        case 'black':
            mask = load_image(mask_path, 'gray')
        case _:
            raise TypeError

    mask = mask.astype(np.float32) / 255.0

    binary = aiccm.get_binary(ori_img)
    binary = (binary == 255).astype(np.float32)  # 255->1.0, 0->0.0
    binary_tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    ori_tensor = torch.from_numpy(ori).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    ori_tensor = ori_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    binary_tensor = binary_tensor.to(device)

    return ori_tensor, mask_tensor, binary_tensor


