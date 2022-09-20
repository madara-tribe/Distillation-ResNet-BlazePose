import sys
sys.path.append('../src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import load_anchors, _decode_boxes

def distilladtion_loss(logits, target, bn_logits, bn_target):
    T = 0.01
    k = 100
    thresh = 100
    criterion = nn.MSELoss()
    # c : preprocess for distillation
    logits1 = logits[1].squeeze(dim=2)
    target1 = target[1].squeeze(dim=2).detach()
    closs = nn.KLDivLoss(reduction="batchmean", log_target=True)(F.log_softmax(logits1, dim=1)/T, F.log_softmax(target1, dim=1)/T) + F.binary_cross_entropy(logits1.sigmoid(), target1.sigmoid()) 
    # r
    anchor = load_anchors("src/anchors_pose.npy")
    rlogits = _decode_boxes(logits[0], anchor)
    rtarget = _decode_boxes(target[0], anchor)
    rloss = criterion(rlogits, rtarget) 
     
    # backbone
    bn_loss = criterion(bn_logits, bn_target)
    return closs+rloss, bn_loss


