import numpy as np
import os, torch, torchvision, random
import torchvision.transforms.functional as ttf
from torchvision.transforms import InterpolationMode
from timm.models.layers import to_2tuple


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    hflip = hflip and (split == 'train' and random.random() < 0.5)                         # horizontal flip OR rotate
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] > 3:                                                                   # some images have 4 channels
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()     # HWC to CHW
    img = img*(min_max[1] - min_max[0]) + min_max[0]                                       # to range min_max
    return img


hflip = torchvision.transforms.RandomHorizontalFlip()                                      # 数据增强
def transform_augment(imgs, phase='val', min_max=(0, 1)):
    if phase == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
