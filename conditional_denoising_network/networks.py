import torch.nn as nn
from torch.nn import init
from torch.nn import modules
import functools, logging, torch
from model.initial_predictor.modelsBN import *
logger = logging.getLogger('base')


def weights_init_normal(m, std=0.02):                                                      # 使用随机初始化模型的参数
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:                                                       # 对卷积层的初始化
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:                                                   # 对线性层的初始化
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:                                              # 对批量归一化层的初始化
        init.normal_(m.weight.data, 1.0, std)                                              # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):                                                      # 使用凯明随机初始化模型的参数
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:                                                     # 对卷积层的初始化
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:                                                   # 对线性层的初始化
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:                                              # 对批量归一化层的初始化
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


# nn.init.orthogonal_(): 初始化模型参数, 会在高斯分布上采样, 还会对采样的矩阵进行正交化, 使其满足正交矩阵的性质
def weights_init_orthogonal(m):                                                            # 使用正交随机初始化模型的参数
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:                                                       # 对卷积层的初始化
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:                                                   # 对线性层的初始化
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:                                              # 对批量归一化层的初始化
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):                             # 根据配置文件对模型的参数进行初始化
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        # functools.partial(): 创建一个新的函数, 该函数是对原函数的某些参数进行了固定, 以便在调用时不必每次都传递这些参数
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


def define_G(opt):                                                                         # 生成器
    model_opt = opt['model']
    from model.conditional_denoising_network import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups'] = 32

    model = unet.UNet(                                                                     # 定义SR3中的U-Net部分
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )

    predictor = GeneratorResNet(
        input_channels=model_opt['resnet']['input_channels'],
        num_residual_blocks=model_opt['resnet']['num_residual_blocks']
    )

    netG = diffusion.GaussianDiffusion(                                                    # 定义SR3的模型结构
        model,
        predictor,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',                                                                    # 设置损失: L1 or L2
        conditional=model_opt['diffusion']['conditional'],                                 # 是否使用条件去噪扩散概率模型
        schedule_opt=model_opt['beta_schedule']['train']
    )

    if opt['phase'] == 'train':                                                            # 调用初始化方法对模型参数进行赋值
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)                                                       # 设置并行运行模型

    return netG
