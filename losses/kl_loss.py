import sys
sys.path.append('../src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_anchors, _decode_boxes

def distilladtion_loss(logits, target, bn_logits, bn_target):
    T = 0.01
    alpha = 0.6
    thresh = 100
    criterion = nn.SmoothL1Loss()
    backbone_criterion = nn.MSELoss()
    # c : preprocess for distillation
    logits1 = logits[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    target1 = target[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1).detach()
    closs = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((logits1 / T), dim = 1), F.softmax((target1 / T), dim = 1))*(alpha * T * T) + F.binary_cross_entropy(logits1, target1) * (1-alpha)
    
    # r
    anchor = load_anchors("src/anchors_pose.npy")
    rlogits = _decode_boxes(logits[0], anchor)
    rtarget = _decode_boxes(target[0], anchor)
    rloss = criterion(rlogits, rtarget) 
     
    # backbone
    bn_loss = backbone_criterion(bn_logits, bn_target)
    return closs, rloss, bn_loss


def alternative_kl_loss(logits, target):
    T = 0.01
    alpha = 0.6
    thresh = 100
    criterion = nn.L1Loss()
    # c : preprocess for distillation
    log2div = logits[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    tar2div = target[1].clamp(-thresh, thresh).sigmoid().squeeze(dim=-1)
    closs = nn.KLDivLoss(reduction="batchmean")(F.log_softmax((log2div / T), dim = 1), F.softmax((tar2div / T), dim = 1))*(alpha * T * T) + criterion(log2div, tar2div) * (1-alpha)
    
    # r
    anchor = load_anchors("src/anchors.npy")
    rlogits = decode_boxes(logits[0], anchor)
    rtarget = decode_boxes(target[0], anchor)
    rloss = criterion(rlogits, rtarget)
     
    return closs + rloss

