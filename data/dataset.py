import os, random, torch
from PIL import Image
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from data.util import transform_augment


class MyDataset(Dataset):                                                                  # 训练数据集
    def __init__(self, inp_train_path, tar_train_path, patch_size=64, data_len=-1, phase='train'):
        super().__init__()

        self.inputPath = inp_train_path
        self.inputImages = os.listdir(inp_train_path)                                      # 输入图片路径下的所有文件名列表
        self.targetPath = tar_train_path
        self.targetImages = os.listdir(tar_train_path)                                     # 目标图片路径下的所有文件名列表
        self.ps = patch_size
        self.phase = phase
        self.data_len = data_len                                                           # 这个值为 -1 的时候就是加载全部数据集

    def __len__(self):
        return len(self.targetImages[0:self.data_len])                                     # 指定要加载数据的长度

    def __getitem__(self, index):
        ps = self.ps
        index = index % len(self.targetImages[0:self.data_len])                            # 取模，防止索引溢出

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])             # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')                             # 读取图片
        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)                                             # 将图片转为张量
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]                                # 图片的高和宽
        input_ = inputImage                                                                # 裁剪patch ，输入和目标patch要对应相同
        target = targetImage

        if self.phase == 'train':
            if ps:
                rr = random.randint(0, hh - ps)                                            # 随机数： patch 左下角的坐标 (rr, cc)
                cc = random.randint(0, ww - ps)
                input_ = inputImage[:, rr:rr+ps, cc:cc+ps]                                 # 裁剪 patch ，输入和目标 patch 要对应相同
                target = targetImage[:, rr:rr+ps, cc:cc+ps]

        # 数据增强
        [input_, target] = transform_augment([input_, target], phase=self.phase, min_max=(-1, 1))
        return {'Target': target, 'Input': input_, 'Index': index}, self.inputImages[index]