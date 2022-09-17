# Distillation of blazepose

# abstract 

This is BlazePose Distillation to student model for being lighten. And for getting more clear performance of teacher model.

<b>This is overall stracture</b>


# Hint-Based Distillation
This is kind of Distillation that use teacher model backbon(embedding) output for Distillation.

```python 
# Distillation loss
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
```



# Distillation Performance (student model)

## teacher model / student model


## loss curve
















# dataset
- [MPII Human Pose Models](https://pose.mpi-inf.mpg.de)
- [VGG Human Pose Estimation datasets](https://www.robots.ox.ac.uk/~vgg/data/pose/)

# References
- [tflite models to PyTorch](https://github.com/zmurez/MediaPipePyTorch)
