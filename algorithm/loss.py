import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from model.arcface import *

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1),reduce=reduce)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def loss_jocor(y_1, y_2, arcLoss1, arcLoss2, forget_rate, ind, co_lambda1, co_lambda2, co_lambda, co_lambda_constant,
               reduce=False):
    if co_lambda_constant:
        co_lambda1 = co_lambda
        co_lambda2 = co_lambda

    loss_pick = (arcLoss1 * (1 - co_lambda1) + arcLoss2 * (1 - co_lambda2) + (
            co_lambda1 + co_lambda2) * 0.5 * kl_loss_compute(y_1, y_2, reduce=reduce) + (
                         co_lambda1 + co_lambda2) * 0.5 * kl_loss_compute(y_2, y_1, reduce=reduce)).cpu()
    # loss_pick = (0.5 * loss_pick_1 + 0.5 * loss_pick_2).cpu()
    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ######arcface######
    # pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/(float(num_remember))
    pure_ratio = np.sum(ind[ind_sorted[:num_remember]])/float(num_remember)
    ind_update = ind_sorted[:num_remember]
    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio

def loss_jocor_tri(y_1, y_2, y_3, arcLoss1, arcLoss2, arcLoss3, forget_rate, ind, co_lambda1, co_lambda2, co_lambda3,
                   co_lambda, co_lambda_constant, reduce=False):
    # loss_pick = (arcLoss1 * (1 - co_lambda1) + arcLoss2 * (1 - co_lambda2) + arcLoss3 * (1 - co_lambda3) +
    #              (co_lambda1 + co_lambda2 + co_lambda3) * 0.25 * kl_loss_compute(y_1, y_2, reduce=reduce) +
    #              (co_lambda1 + co_lambda2 + co_lambda3) * 0.25 * kl_loss_compute(y_2, y_1, reduce=reduce) +
    #              (co_lambda1 + co_lambda2 + co_lambda3) * 0.25 * kl_loss_compute(y_3, y_1, reduce=reduce) +
    #              (co_lambda1 + co_lambda2 + co_lambda3) * 0.25 * kl_loss_compute(y_3, y_2, reduce=reduce)).cpu()
    if co_lambda_constant:
        co_lambda1 = co_lambda
        co_lambda2 = co_lambda
        co_lambda3 = co_lambda

    loss_pick = ((1-co_lambda1)*arcLoss1 + (1-co_lambda2)*arcLoss2 + (1-co_lambda3)*arcLoss3 +
                 co_lambda1*kl_loss_compute(y_1, y_2, reduce=reduce) +
                 co_lambda2*kl_loss_compute(y_2, y_1, reduce=reduce) +
                 co_lambda3/2 * (kl_loss_compute(y_3, y_1, reduce=reduce) + kl_loss_compute(y_3, y_2, reduce=reduce))).cpu()
    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ######arcface######
    pure_ratio = np.sum(ind[ind_sorted[:num_remember]])/float(num_remember)
    ind_update = ind_sorted[:num_remember]
    loss = torch.mean(loss_pick[ind_update])
    return loss, loss, loss, pure_ratio, pure_ratio, pure_ratio
