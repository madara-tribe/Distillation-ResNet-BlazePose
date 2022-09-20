import os, sys
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from fastprogress import master_bar, progress_bar
from utils.optimizer import create_optimizer
from utils.DataLoader import data_loader, call_data_loader
from cfg import Cfg
from losses import utils, kl_loss
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

class Trainer:
    def __init__(self, config, device, num_workers=1):
        self.writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
        print('loading dataloader....')
        train_dst, val_dst, test_dst = get_dataset(config)
        self.train_loader = call_data_loader(train_dst, bs=config.bs, num_worker=num_workers)
        self.val_loader = call_data_loader(val_dst, bs=config.val_bs, num_worker=0)
        self.test_loader = call_data_loader(test_dst, bs=1, num_worker=0)
        self.attn_layer = utils.SelfAttntion(256).to(device)
        self.cur_itrs = 0
        #self.best_val_loss = 100.0

    def train(self, c, device, student_weights=None):
        teacher_detector, _ = load_blazepose(device, teacher=True)
        student_detector, _ = load_blazepose(device, teacher=False, weight=student_weights)
        teacher_detector.eval()
        model = student_detector
        
        
        print('load model && set parameter')
        optimizer, scheduler = create_optimizer(model, c)
        total_itrs = c.total_itrs

        while self.cur_itrs < total_itrs:
            start_time = time.time()
            model.train()
            avg_rcloss = avg_bnloss = 0.              
            for images in progress_bar(self.train_loader):
                self.cur_itrs += 1
                x_batch = images.to(device, dtype=torch.float32)
        #        print(x_batch.shape, x_batch.min(), x_batch.max())
                optimizer.zero_grad()
                lesson, backborn_lesson = teacher_detector(x_batch)
                logits, backborn_logits = model(x_batch)
                bn_lesson = self.attn_layer(backborn_lesson[1])
                bn_logits = self.attn_layer(backborn_logits[1])
                rcloss, bn_loss = kl_loss.distilladtion_loss(logits, lesson, bn_logits, bn_lesson) 
                loss = rcloss + bn_loss
                loss.backward()
                optimizer.step()

                avg_rcloss += rcloss.item() 
                avg_bnloss += bn_loss.item()
                if self.cur_itrs % c.freq==0:
                    print('avg_rcloss, avg_bnloss', avg_rcloss/c.freq, avg_bnloss/c.freq)
                    self.writer.add_scalar('train/avg_rcLoss', avg_rcloss/c.freq, self.cur_itrs) 
                    self.writer.add_scalar('train/avg_cLoss', avg_bnloss/c.freq, self.cur_itrs)
                    avg_rcloss = avg_bnloss = 0.
            
                if self.cur_itrs % c.val_freq==0:
                    model, teacher_detector = self.validation(model, teacher_detector)
                if self.cur_itrs % c.test_freq==0: 
                    model, teacher_detector = self.test(model, teacher_detector)
                if self.cur_itrs > total_itrs:
                    break
                model.train()
                scheduler.step()
            self.writer.close()

    def validation(self, model, teacher_detector):
        avg_val_rcloss = avg_val_bnloss = 0
        model.eval()
        for idx, val_batch in tqdm(enumerate(self.val_loader)):
            val_batch = val_batch.to(device, dtype=torch.float32)
            ## validation kl_divergence
            val_lesson, val_bn_lesson = teacher_detector(val_batch)
            val_logits, val_bn_logits = model(val_batch)
            val_bn_lesson = self.attn_layer(val_bn_lesson[1])
            val_bn_logits = self.attn_layer(val_bn_logits[1])
            val_rcloss, val_bn_loss = kl_loss.distilladtion_loss(val_logits, val_lesson, val_bn_logits, val_bn_lesson)
            avg_val_rcloss += val_rcloss.item() / len(self.val_loader)
            avg_val_bnloss += val_bn_loss.item() / len(self.val_loader)
        print('val_loss', avg_val_rcloss, avg_val_bnloss)
        avg_val_loss = avg_val_rcloss + avg_val_bnloss
        self.writer.add_scalar('valid/avg_rcloss', avg_val_rcloss, self.cur_itrs)
        self.writer.add_scalar('valid/avg_bnloss', avg_val_bnloss, self.cur_itrs)
        #if avg_val_loss < self.best_val_loss:
            #self.best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'checkpoints/student_iter{}.pth'.format(self.cur_itrs))
        print('succeess to checkpoints/student_iter{}.pth'.format(self.cur_itrs))
        model.train()
        return model, teacher_detector

    def test(self, model, teacher_detector):
        model.eval()
        avg_rc_error = avg_bn_error = 0
        for test_batch in self.test_loader:
            test_batch = test_batch.to(device, dtype=torch.float32)
            test_lesson, test_bn_lesson = teacher_detector(test_batch)
            test_logits, test_bn_logits = model(test_batch)
            test_bn_lesson = self.attn_layer(test_bn_lesson[1])
            test_bn_logits = self.attn_layer(test_bn_logits[1])
            rc_error, bn_error = kl_loss.distilladtion_loss(test_logits, test_lesson, test_bn_lesson, test_bn_logits)
            avg_rc_error += rc_error.item() / len(self.test_loader)
            avg_bn_error += bn_error.item() / len(self.test_loader)
        print('RC error, bn error', avg_rc_error, avg_bn_error)
        self.writer.add_scalar('test/avg_rc_error', avg_rc_error, self.cur_itrs)
        self.writer.add_scalar('test/avg_bn_error',avg_bn_error, self.cur_itrs)
        model.train()
        return model, teacher_detector
                

if __name__=='__main__':
    if len(sys.argv) > 1:
        student_weights = str(sys.argv[1])
    else:
        student_weights = None
    cfg = Cfg
    os.makedirs(cfg.checkpoints, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(cfg, device, num_workers=1)
    trainer.train(c=cfg,
          device=device,
          student_weights = student_weights)


