import random, torch
from torch.autograd import Variable                                   # torch中Variable模板


class ReplayBuffer:                                                   #
    def __init__(self, max_size=50):
        # assert 表达式，字符串 --> 当表达式为True时，什么反应都没有；当表达式为False时，输出后面这个字符串
        assert max_size > 0, "空的缓冲区，要小心！"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)                     # 对数据维度进行扩充  [C,H,W]->[N,C,H,W]
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:                        # 根据四舍五入，得到一个在[0，1)或[0，1]范围内的随机数
                    i = random.randint(0, self.max_size - 1)          # randint产生的随机数是在指定的某个区间内的一个值
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:                                                       # 定义学习率衰减方式
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs-decay_start_epoch) > 0, "衰减必须在训练结束前开始!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0-max(0,epoch+self.offset-self.decay_start_epoch)/(self.n_epochs-self.decay_start_epoch)