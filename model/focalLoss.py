import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None,ignore_index=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):

        # input = [batch_size * max_len] * label_num

        input = input[target != self.ignore_index]
        target = target[target != self.ignore_index]

        target = target.view(-1, 1)

        pt = F.softmax(input,dim = -1)

        pt = pt.gather(1,target)
        pt = pt.view(-1)

        logpt = torch.log(pt)


        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


