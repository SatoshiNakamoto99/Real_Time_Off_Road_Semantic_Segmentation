
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)



def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    # Assicurati che x e y siano tensori torch
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # Trova la device comune
    device = x.device if x.is_cuda else (y.device if y.is_cuda else torch.device('cpu'))
    
    # Trasferisci tutti i tensori sulla stessa device
    x = x.to(device)
    y = y.to(device)
    
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        mask = mask.to(device)
    
    # Crea la maschera k
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask)
    
    # Verifica se ci sono elementi validi
    if not k.any():
        return np.zeros((n, n), dtype=int)  # Restituisci una matrice di zeri se non ci sono elementi validi
    
    # Converte x e y in interi prima di calcolare il bincount
    x_int = x[k].to(torch.int64)  # Converte a int64
    y_int = y[k].to(torch.int64)  # Converte a int64

    # Calcola la matrice di confusione utilizzando np.bincount su tensori CPU
    # Usa un valore di offset se necessario
    bincount = np.bincount((n * x_int.cpu().numpy() + y_int.cpu().numpy()).astype(int), minlength=n**2)

    return bincount.reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calcola l'accuratezza globale
        globalacc = np.diag(conf_matrix).sum() / np.float64(conf_matrix.sum())
        
        # Calcola la precisione per tutte le classi
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float64)
        
        # Verifica che la precisione non sia NaN o Inf
        if np.isnan(classpre[1]) or np.isinf(classpre[1]):
            pre = 0
        else:
            pre = classpre[1]
        
        # Calcola il richiamo per tutte le classi
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float64)
        
        # Verifica che il richiamo non sia NaN o Inf
        if np.isnan(classrecall[1]) or np.isinf(classrecall[1]):
            recall = 0
        else:
            recall = classrecall[1]
        
        # Calcola l'Intersezione sopra l'unione (IoU) per tutte le classi
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float64)
        
        # Seleziona le metriche per la classe positiva (classe 1)
        iou = IU[1]
        
        # Calcola l'F-score
        if pre + recall == 0:
            F_score = 0
        else:
            F_score = 2 * (recall * pre) / (recall + pre)

    return globalacc, pre, recall, F_score, iou, IU[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train_sensor_fusion(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    
    total_loss = 0
    
    for i, (_, _,rgb,depth, target) in pbar:
        if args.onGPU:
                rgb = rgb.cuda().float() 
                depth = depth.cuda().float()
        
        output = model(rgb,depth)
        
        optimizer.zero_grad()
        focal_loss, tversky_loss, loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) % 
                             (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))
    
    avg_loss = total_loss / total_batches
    return avg_loss

def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    
    total_loss = 0
    
    for i, (_, input, target) in pbar:
        if args.onGPU:
                input = input.cuda().float() 
        
        output = model(input)
        
        optimizer.zero_grad()
        focal_loss, tversky_loss, loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) % 
                             (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))
    
    avg_loss = total_loss / total_batches
    return avg_loss


def train16fp(args, train_loader, model, criterion, optimizer, epoch,scaler):
    model.train()
    print("16fp-------------------")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))

@torch.no_grad()
def val_sensor_fusion(val_loader, model, criterion):
    model.eval()

    DA = SegmentationMetric(2)
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    conf_mat = np.zeros((2,2), dtype=np.float64)
    
    total_loss = 0

    for i, (_, _,rgb,depth, target) in pbar:
        
        rgb = rgb.cuda().float()
        depth = depth.cuda().float()

        # run the model
        output = model(rgb,depth)

        # calculate loss
        focal_loss, tversky_loss, loss = criterion(output, target)
        total_loss += loss.item()

        out_da = output
        target_da = target

        _, da_predict = torch.max(out_da, 1)
        _, da_gt = torch.max(target_da, 1)

        conf_mat += confusion_matrix(da_gt.cpu().numpy(), da_predict.cpu().numpy(), 2)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc, rgb.size(0))
        da_IoU_seg.update(da_IoU, rgb.size(0))
        da_mIoU_seg.update(da_mIoU, rgb.size(0))

    globalacc, pre, recall, F_score, iou , iou_0= getScores(conf_mat)
    
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    da_segment_result_OFFNET = (globalacc, pre, recall, F_score, iou, iou_0)
    
    avg_loss = total_loss / total_batches
    
    return da_segment_result, da_segment_result_OFFNET, avg_loss


@torch.no_grad()
def val(val_loader, model, criterion):
    model.eval()

    DA = SegmentationMetric(2)
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    conf_mat = np.zeros((2,2), dtype=np.float64)
    
    total_loss = 0

    for i, (_, input, target) in pbar:
        
        input = input.cuda().float()

        # run the model
        output = model(input)

        # calculate loss
        focal_loss, tversky_loss, loss = criterion(output, target)
        total_loss += loss.item()

        out_da = output
        target_da = target

        _, da_predict = torch.max(out_da, 1)
        _, da_gt = torch.max(target_da, 1)

        conf_mat += confusion_matrix(da_gt.cpu().numpy(), da_predict.cpu().numpy(), 2)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc, input.size(0))
        da_IoU_seg.update(da_IoU, input.size(0))
        da_mIoU_seg.update(da_mIoU, input.size(0))

    globalacc, pre, recall, F_score, iou , iou_0= getScores(conf_mat)
    
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    da_segment_result_OFFNET = (globalacc, pre, recall, F_score, iou, iou_0)
    
    avg_loss = total_loss / total_batches
    
    return da_segment_result, da_segment_result_OFFNET, avg_loss




def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

