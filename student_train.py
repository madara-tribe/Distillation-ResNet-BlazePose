import os, sys
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.DataLoader import data_loader, call_data_loader
from cfg import Cfg
from losses.kl_loss import kl_divergence_loss
from teacher.blazebase import resize_pad, denormalize_detections
from teacher.blazepose import BlazePose as tBlazePose
from teacher.blazepose_landmark import BlazePoseLandmark as tBlazePoseLandmark

from student.blazepose import BlazePose as sBlazePose
from student.blazepose_landmark import BlazePoseLandmark as sBlazePoseLandmark
from utils.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def train(config, device, num_workers, student_weights=None, epochs=10):
    teacher_detector, teacher_regressor = load_blazepose(device, teacher=True)
    student_detector, student_regressor = load_blazepose(device, teacher=False, weight=student_weights)
    
    teacher_detector.eval()
    teacher_regressor.eval()
    student_detector.train()
    student_regressor.eval()
   

    print('loading dataloader....')
    train_dst, val_dst, test_dst = get_dataset(config)
    train_loader = call_data_loader(train_dst, bs=config.bs, num_worker=num_workers)
    val_loader = call_data_loader(val_dst, bs=config.val_bs, num_worker=0)
    test_loader = call_data_loader(test_dst, bs=1, num_worker=0)
    
    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
    
    print('load mdoel && set parameter')
    acc_criterion = nn.MSELoss()    
    model = student_detector
    optimizer, scheduler = create_optimizer(model, config)
    cur_itrs = 0
    total_itrs = config.total_itrs
    """
    while cur_itrs < total_itrs:
        start_time = time.time()
        model.train()
        avg_loss = 0.              
        for images in progress_bar(train_loader):
            cur_itrs += 1
            x_batch = images.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            lesson = teacher_net(x_batch)
            logits = model(x_batch)
            
            loss = kl_divergence_loss(logits, lesson) 
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() 
            if cur_itrs % 100==0:
                print('avg_loss', avg_loss/100)
                writer.add_scalar('train/avg_Loss', avg_loss, cur_itrs) 
                avg_loss = 0.
          
            if cur_itrs % 4000==0:
                avg_val_loss = 0
                model.eval()
                for idx, val_batch in tqdm(enumerate(val_loader)):
                    val_batch = val_batch.to(device, dtype=torch.float32)
                    ## validation kl_divergence
                    val_lesson = teacher_net(val_batch)
                    val_logits = model(val_batch)
                    val_loss = kl_divergence_loss(val_logits, val_lesson)
                    avg_val_loss += val_loss.item() / len(val_loader)
                     
                print('val_loss', avg_val_loss)
                writer.add_scalar('valid/avg_loss', avg_val_loss, cur_itrs)
                
                torch.save(model.state_dict(), 'checkpoints/student_iter{}.pth'.format(cur_itrs))
                print('succeess to checkpoints/student_iter{}.pth'.format(cur_itrs))
                model.train()
            
            if cur_itrs % 1000==0: 
                model.eval()
                avg_rmae_acc = 0 
                avg_cmae_acc = 0
                for test_batch in test_loader:
                    test_batch = test_batch.to(device, dtype=torch.float32)
                    test_lesson = teacher_net(test_batch)
                    test_logits = model(test_batch)
                    mae_rscore = acc_criterion(test_logits[0], test_lesson[0])
                    mae_cscore = acc_criterion(test_logits[1], test_lesson[1])
                    avg_rmae_acc += mae_rscore / len(test_loader)
                    avg_cmae_acc += mae_cscore / len(test_loader)
                print('R mae accuracy', avg_rmae_acc)
                print('C mae accuracy', avg_cmae_acc) 
                writer.add_scalar('test/avg_Rmae_acc', avg_rmae_acc, cur_itrs)
                writer.add_scalar('test/avg_Cmae_acc', avg_cmae_acc, cur_itrs)
                model.train()
            if cur_itrs > total_itrs:
                break
            model.train()
            scheduler.step()
        writer.close()
    """
if __name__=='__main__':
    if len(sys.argv) > 1:
        student_weights = str(sys.argv[1])
    else:
        student_weights = None
    cfg = Cfg
    os.makedirs(cfg.checkpoints, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(config=cfg,
          device=device,
          student_weights = student_weights,
          num_workers=0)
