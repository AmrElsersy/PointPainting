"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse, cv2
import logging
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

from model.BiseNetv2 import BiSeNetV2
from model.ohem_loss import OhemCELoss
from dataset import create_train_dataloader, create_val_dataloader
from utils.utils import *
from visualization import KittiVisualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoints/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoints/logging', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data/KITTI', help='root path of dataset')
    parser.add_argument('--pretrained', type=str,default='checkpoints/BiseNetv2.pth',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoints', help='save checkpoint path')    
    parser.add_argument('--savefreq', type=int, default=1, help="save weights each freq num of epochs")
    parser.add_argument('--logdir', type=str, default='checkpoints/logging', help='logging')    
    parser.add_argument("--lr_patience", default=40, type=int)
    args = parser.parse_args()
    return args
# ======================================================================

torch.cuda.empty_cache()

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

args = parse_args()
logging.basicConfig( format='[%(message)s', level=logging.INFO,
handlers=[logging.FileHandler(args.logdir, mode='w'), logging.StreamHandler()])
writer = tensorboard.SummaryWriter(args.tensorboard)
visualizer = KittiVisualizer()

def main():
    # ========= dataloaders ===========
    train_dataloader = create_train_dataloader(root=args.datapath, batch_size=args.batch_size)
    val_dataloader = create_val_dataloader(root=args.datapath, batch_size=args.batch_size)

    start_epoch = 0

    # ======== models & loss ========== 
    model = BiSeNetV2(n_classes=19, output_aux=True)
    loss = OhemCELoss(0.7)
    loss_aux = [OhemCELoss(0.7) for _ in range(4)]

    # ========= load weights ===========
    if args.resume:
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['bisenetv2'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\tLoaded checkpoint from {args.pretrained}\n')
        time.sleep(1)
    else:
        print("******************* Start training from scratch *******************\n")
        # time.sleep(2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # =========== train / validate ===========
        train_loss = train_one_epoch(model, loss, loss_aux, optimizer, train_dataloader, epoch)
        val_loss = validate(model, loss, val_dataloader, epoch)
        scheduler.step(val_loss)
        logging.info(f"\ttraining epoch={epoch} .. train_loss={train_loss}")
        logging.info(f"\tvalidation epoch={epoch} .. val_loss={val_loss}")
        time.sleep(2)
        # ============= tensorboard =============
        writer.add_scalar('train_loss',train_loss, epoch)
        writer.add_scalar('val_loss',val_loss, epoch)
        # ============== save model =============
        if epoch % args.savefreq == 0:
            checkpoint_state = {
                'bisenetv2': model.state_dict(),
                "epoch": epoch
            }
            savepath = os.path.join(args.savepath, f'weights_epoch_{epoch}.pth.tar')
            torch.save(checkpoint_state, savepath)
            print(f'\n\t*** Saved checkpoint in {savepath} ***\n')
            time.sleep(2)
    writer.close()

def train_one_epoch(model, criterion, criterion_aux, optimizer, dataloader, epoch):
    model.train()
    model.to(device)
    losses = []

    for images, labels in tqdm(dataloader):
        images = images.to(device) # (batch, 3, H, W)
        labels = labels.to(device) # (batch, H, W) 

        image = images[0].cpu().detach().numpy().transpose(1,2,0)
        cv2.imshow('f', image)
        image = images[1].cpu().detach().numpy().transpose(1,2,0)
        cv2.imshow('f1', image)
        cv2.waitKey(0)

        logits, *logits_aux = model(images) # (batch, 19, 1024, 2048)

        loss_main = criterion(logits, labels)
        loss_aux = [crit(lgt, labels) for crit, lgt in zip(criterion_aux, logits_aux)]
        loss = loss_main + sum(loss_aux)

        losses.append(loss.cpu().item())
        print(f'training @ epoch {epoch} .. loss = {round(loss.item(),3)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return round(np.mean(losses).item(),3)


def validate(model, criterion, dataloader, epoch):
    model.eval()
    model.to(device)
    losses = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device) # (batch, 3, H, W)
            labels = labels.to(device) # (batch, H, W) 

            logits, *logits_aux = model(images) # (batch, 19, 1024, 2048)

            loss = criterion(logits, labels)
            losses.append(loss.cpu().item())

            print(f'validation @ epoch {epoch} .. loss = {round(loss.item(),3)}')
        return round(np.mean(losses).item(),3)

if __name__ == "__main__":
    main()