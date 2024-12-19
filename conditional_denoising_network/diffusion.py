import numpy as np
from tqdm import tqdm
import torch, lpips, warnings
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from facenet_pytorch import InceptionResnetV1
warnings.filterwarnings("ignore")
lpips_model = lpips.LPIPS(net="vgg").eval().to('cuda')                                     # 加载预训练的LPIPS模型, 抽取特征的网络可以有['alex', 'vgg']
vggface = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')                       # 加载预训练的人脸识别模型


# 根据配置文件选择构建beta的方法
def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'linear':                                                               # 线性产生beta
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    else:
        raise NotImplementedError(schedule)
    return betas


def exists(x):                                                                             # 定义一个判断是否存在的方法
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):                                                        # 高斯扩散模型的类
    def __init__(self, denoise_fn, predictor, image_size, channels=3, loss_type='l1', conditional=True, schedule_opt=None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.predictor = predictor
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):                                                            # 定义损失函数
        self.loss_L1 = nn.L1Loss(reduction='sum').to(device)
        self.loss_L2 = nn.MSELoss(reduction='sum').to(device)
        self.loss_SL1 = nn.SmoothL1Loss(reduction='sum', beta=1.0).to(device)

    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1)/(torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def set_new_noise_schedule(self, schedule_opt, device):                                # 初始化扩散模型的各个参数
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(schedule=schedule_opt['schedule'], n_timestep=schedule_opt['n_timestep'], linear_start=schedule_opt['linear_start'], linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas #  isinstance(): 来判断一个对象是否是一个已知的类型
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)                                        # np.cumprod(): 按axis轴计算累加和
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # self.register_buffer('name',Tensor): 用于定义一组参数. 该组参数在模型训练时不会更新（即调用optimizer.step()后该组参数不会变化, 只可人为地改变它们的值）
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 计算先验分布: q(x_t | x_{t-1}) 和其他
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)    # 计算后验分布的方差
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # 由于扩散链开始时的后验方差为 0，使用对数计算进行剪切
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):                                     # 由x_t推导出x_0: x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):                                                # 求解出后验概率的均值: mu_mean = np.sqrt(alpha) * (1 - alpha_bar_prev) / (1 -alpha_bar) * x_t + beta * np.sqrt(alpha_bar_prev) / (1 - alpha_bar) * x_0
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]            # 后验分布的方差进行裁剪
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):                # 供外部调用求解后验分布均值和方差的方法
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    # 装饰器: @torch.no_grad()中的数据不需要计算梯度, 也不会进行方向传播
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):                        # 反向去噪过程
        model_mean, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        x = x_in
        shape = x.shape
        img = torch.randn(shape, device=device)
        ret_img = x
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, i, condition_x=x)
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):                                     # 超分重建过程
        # return self.p_sample_loop(self.predictor(x_in), continous) + self.predictor(x_in)
        return self.predictor(x_in)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):                # 前向加噪过程
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    def p_losses(self, x_in, noise=None):                                                  # 计算损失的方法
        predicted_img = self.predictor(x_in['Input'])
        x_start = x_in['Target'] - predicted_img
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 前向加噪过程
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        # 反向去噪过程(条件去噪扩散概率模型)
        x_recon = self.denoise_fn(torch.cat([predicted_img, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # 预测的残差图
        x_residual = self.predict_start_from_noise(x_noisy, t=t, noise=x_recon)

        # 身份一致性损失
        emb_target = vggface(F.interpolate(x_in['Target'], (112, 112), mode='bilinear', align_corners=True))
        emb_source = vggface(F.interpolate(predicted_img + x_residual, (112, 112), mode='bilinear', align_corners=True))
        loss_id = (1 - self.cosin_metric(emb_source, emb_target)).mean()

        # 初始人脸重建损失
        loss_rec = 0.8 * lpips_model(predicted_img + x_residual, x_in['Target']) + 1.2 * self.loss_L1(predicted_img, x_in['Target'])

        # 去噪损失
        loss_deno = self.loss_L1(noise, x_recon)

        loss_total = 1 * loss_id + 1 * loss_rec + 10 * loss_deno        # DDCM_original
        return loss_total

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)