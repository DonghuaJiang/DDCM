import logging
import torch.utils.data
from data.dataset import MyDataset

def create_dataloader(dataset, dataset_opt):                                               # 创建数据加载器的方法
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_opt['batch_size'],
        shuffle=dataset_opt['use_shuffle'],
        num_workers=dataset_opt['num_workers'],
        pin_memory=True)


def create_dataset(dataset_opt, phase='train'):                                            # 创建数据集的方法
    dataset = MyDataset(inp_train_path=dataset_opt['inp_path'],
                        tar_train_path=dataset_opt['tar_path'],
                        patch_size=dataset_opt['patch_size'],
                        data_len=dataset_opt['data_len'],
                        phase=phase)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
