import os, sys
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.optim as optimizers
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.DataLoader import data_loader, call_data_loader
from cfg import Cfg
from losses.kl_loss import distilladtion_loss
from models.teacher.blazepose import BlazePose as tBlazePose
from models.teacher.blazepose_landmark import BlazePoseLandmark as tBlazePoseLandmark
from models.student.blazepose import BlazePose as sBlazePose
from models.student.blazepose_landmark import BlazePoseLandmark as sBlazePoseLandmark

def load_blazepose(device, teacher=True, weight=None):
    if teacher:
        pose_detector = tBlazePose().to(device)
        pose_detector.load_weights("src/blazepose.pth")
        pose_detector.load_anchors("src/anchors_pose.npy")

        pose_regressor = tBlazePoseLandmark().to(device)
        pose_regressor.load_weights("src/blazepose_landmark.pth")
        return pose_detector, pose_regressor
    else:
        pose_detector = sBlazePose().to(device)
        if weight:
            pose_detector.load_weights(weight)
        pose_detector.load_anchors("src/anchors_pose.npy")

        pose_regressor = tBlazePoseLandmark().to(device)
        pose_regressor.load_weights("src/blazepose_landmark.pth")
        return pose_detector, pose_regressor


def get_dataset(config):
    """ Dataset And Augmentation
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    train_dst = data_loader(image_dir=config.train_dir, width=config.width,
                           height=config.height, transform=train_transform)
    val_dst = data_loader(image_dir=config.valid_dir, width=config.width,
                         height=config.height, transform=val_transform)
    test_dst = data_loader(image_dir=config.test_dir, width=config.width,
                         height=config.height, transform=val_transform)
    return train_dst, val_dst, test_dst


def create_optimizer(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min)
    return optimizer, scheduler

def train(c, device, num_workers, student_weights=None, epochs=10):
    teacher_detector, _ = load_blazepose(device, teacher=True)
    student_detector, _ = load_blazepose(device, teacher=False, weight=student_weights)
    teacher_detector.eval()
    model = student_detector
   
    print('loading dataloader....')
    train_dst, val_dst, test_dst = get_dataset(c)
    train_loader = call_data_loader(train_dst, bs=c.bs, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=c.val_bs, num_worker=0)
    test_loader = call_data_loader(test_dst, bs=1, num_worker=0)
    
    writer = SummaryWriter(log_dir=c.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{c.TRAIN_OPTIMIZER}_LR_{c.lr}_BS_Size_{c.width}',
                           comment=f'OPT_{c.TRAIN_OPTIMIZER}_LR_{c.lr}_BS_Size_{c.width}')
    
    print('load mdoel && set parameter')
    optimizer, scheduler = create_optimizer(model, c)
    cur_itrs = 0
    best_val_loss = 100.0
    total_itrs = c.total_itrs

    while cur_itrs < total_itrs:
        start_time = time.time()
        model.train()
        avg_closs = avg_rloss = avg_bnloss = 0.              
        for images in progress_bar(train_loader):
            cur_itrs += 1
            x_batch = images.to(device, dtype=torch.float32)
#            print(x_batch.shape, x_batch.min(), x_batch.max())
            optimizer.zero_grad()
            lesson, backborn_lesson = teacher_detector(x_batch)
            logits, backborn_logits = model(x_batch)
            rloss, closs, bn_loss = distilladtion_loss(logits, lesson, backborn_logits, backborn_lesson) 
            loss = rloss + closs + bn_loss
            loss.backward()
            optimizer.step()

            avg_rloss += rloss.item() 
            avg_closs += closs.item()
            avg_bnloss += bn_loss.item()
            if cur_itrs % c.freq==0:
                print('avg_rloss, avg_closs', avg_rloss/c.freq, avg_closs/c.freq, avg_bnloss/c.freq)
                writer.add_scalar('train/avg_rLoss', avg_rloss/c.freq, cur_itrs) 
                writer.add_scalar('train/avg_cLoss', avg_closs/c.freq, cur_itrs)
                writer.add_scalar('train/avg_cLoss', avg_bnloss/c.freq, cur_itrs)
                avg_closs = avg_rloss = avg_bnloss = 0.
          
            if cur_itrs % c.val_freq==0:
                avg_val_rloss = avg_val_closs = avg_val_bnloss = 0
                model.eval()
                for idx, val_batch in tqdm(enumerate(val_loader)):
                    val_batch = val_batch.to(device, dtype=torch.float32)
                    ## validation kl_divergence
                    val_lesson, val_bn_lesson = teacher_detector(val_batch)
                    val_logits, val_bn_logits = model(val_batch)
                    val_rloss, val_closs, val_bn_loss = distilladtion_loss(val_logits, val_lesson, val_bn_lesson, val_bn_logits)
                    avg_val_rloss += val_rloss.item() / len(val_loader)
                    avg_val_closs += val_closs.item() / len(val_loader)
                    avg_val_bnloss += val_bn_loss.item() / len(val_loader)
                print('val_loss', avg_val_rloss, avg_val_closs, avg_val_bnloss)
                avg_val_loss = avg_val_rloss + avg_val_closs
                writer.add_scalar('valid/avg_rloss', avg_val_rloss, cur_itrs)
                writer.add_scalar('valid/avg_closs', avg_val_closs, cur_itrs)
                writer.add_scalar('valid/avg_bnloss', avg_val_bnloss, cur_itrs)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'checkpoints/student_iter{}.pth'.format(cur_itrs))
                    print('succeess to checkpoints/student_iter{}.pth'.format(cur_itrs))
                model.train()
            
            if cur_itrs % c.test_freq==0: 
                model.eval()
                avg_r_error = avg_c_error = avg_bn_error = 0
                for test_batch in test_loader:
                    test_batch = test_batch.to(device, dtype=torch.float32)
                    test_lesson, test_bn_lesson = teacher_detector(test_batch)
                    test_logits, test_bn_logits = model(test_batch)
                    r_error, c_error, bn_error = distilladtion_loss(val_logits, val_lesson, test_bn_lesson, test_bn_logits)
                    avg_r_error += r_error.item() / len(test_loader)
                    avg_c_error += c_error.item() / len(test_loader)
                    avg_bn_error += bn_error.item() / len(test_loader)
                print('R error, C error, bn error', avg_r_error, avg_c_error, avg_bn_error)
                writer.add_scalar('test/avg_r_error', avg_r_error, cur_itrs)
                writer.add_scalar('test/avg_c_error', avg_c_error, cur_itrs)
                writer.add_scalar('test/avg_bn_error',avg_bn_error, cur_itrs)
                model.train()
            if cur_itrs > total_itrs:
                break
            model.train()
            scheduler.step()
        writer.close()

if __name__=='__main__':
    if len(sys.argv) > 1:
        student_weights = str(sys.argv[1])
    else:
        student_weights = None
    cfg = Cfg
    os.makedirs(cfg.checkpoints, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(c=cfg,
          device=device,
          student_weights = student_weights,
          num_workers=0)

