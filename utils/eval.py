from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['accuracy', 'kl_loss']


def accuracy(output, target, topk=(1.)):
    '''Compute the top1 and top k error'''
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res


def kl_loss(output, target, reverse=False):
    '''after dividing T!
    target is predicted output
    '''
    output_prob = nn.Softmax(-1)(output)
    target_prob = nn.Softmax(-1)(target)
    if reverse:
        loss = -torch.sum(output_prob * torch.log(target_prob / output_prob))
    else:
        loss = -torch.sum(target_prob * torch.log(output_prob / target_prob))

    return loss

def predict(model,dataloader,model_type = 'clf',return_data = True):
    if return_data:
        input_list = []
        target_list = []
    pred_list = []
    for i,(inputs,targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(),targets.cuda()
        inputs, targets = Variable(inputs),Variable(targets)

        outputs = model(inputs)
        if model_type == 'clf':
            outputs = outputs.argmax(1)

            #if return_data:
            #    targets = targets.argmax(1)
        pred_list.append(outputs.data)
        if return_data:
            input_list.append(inputs.data)
            target_list.append(targets.data)

    if return_data:
        return torch.cat(input_list,0), torch.cat(target_list,0),torch.cat(pred_list,0)
    else:
        return torch.cat(pred_list,0)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count
