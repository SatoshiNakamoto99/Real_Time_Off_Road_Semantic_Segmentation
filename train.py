

import os
import torch
import pickle
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, \
    poly_lr_scheduler, val_sensor_fusion, train_sensor_fusion
import torch.optim.lr_scheduler

from loss import TotalLoss

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    if args.sensor_fusion :
        print("Using RGB and Depth")
        if args.adaptive:
            model = net.TwinLiteNet_RGBD_Adaptive()
            if args.encoder_pretrained_depth:
                print(f"Loading pretrained depth encoder from {args.encoder_pretrained_depth}")
                model.rgb_stream.load_state_dict(torch.load(args.encoder_pretrained_depth))
                #freeze the encoder
                if args.freeze_encoder:
                    print("Freezing the encoder")
                    for param in model.rgb_stream.parameters():
                        param.requires_grad = False
            if args.encoder_pretrained_rgb:
                print(f"Loading pretrained rgb encoder from {args.encoder_pretrained_rgb}")
                model.depth_stream.load_state_dict(torch.load(args.encoder_pretrained_rgb))
                #freeze the encoder
                if args.freeze_encoder:
                    print("Freezing the encoder")
                    for param in model.depth_stream.parameters():
                        param.requires_grad = False
        else:
            model = net.TwinLiteNetRGBD()
            if args.encoder_pretrained_depth:
                print(f"Loading pretrained depth encoder from {args.encoder_pretrained_depth}")
                model.encoder_D.load_state_dict(torch.load(args.encoder_pretrained_depth))
                #freeze the encoder
                if args.freeze_encoder:
                    print("Freezing the encoder")
                    for param in model.encoder_D.parameters():
                        param.requires_grad = False
            if args.encoder_pretrained_rgb:
                print(f"Loading pretrained rgb encoder from {args.encoder_pretrained_rgb}")
                model.encoder_RGB.load_state_dict(torch.load(args.encoder_pretrained_rgb))
                #freeze the encoder
                if args.freeze_encoder:
                    print("Freezing the encoder")
                    for param in model.encoder_RGB.parameters():
                        param.requires_grad = False
    elif args.depth:
        print("Using only depth")
        model = net.TwinLiteNet()
        if args.encoder_pretrained_depth:
            print(f"Loading pretrained depth encoder from {args.encoder_pretrained_depth}")
            model.encoder.load_state_dict(torch.load(args.encoder_pretrained_depth))
            if args.freeze_encoder:
                print("Freezing the encoder")
                for param in model.encoder.parameters():
                    param.requires_grad = False
    else:
        print("Using only RGB ")
        model = net.TwinLiteNet()
        if args.encoder_pretrained_rgb:
            print(f"Loading pretrained rgb encoder from {args.encoder_pretrained_rgb}")
            model.encoder.load_state_dict(torch.load(args.encoder_pretrained_rgb))
            if args.freeze_encoder:
                print("Freezing the encoder")
                for param in model.encoder.parameters():
                    param.requires_grad = False
    
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        model_path = args.pretrained
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    else:
        print("Training from scratch.")

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(train_path = args.train_path, valid=False,
                                rgb_folder_name=args.rgb_folder_name,
                                depth_folder_name=args.depth_folder_name,
                                label_folder_name=args.label_folder_name,
                                sensor_fusion=args.sensor_fusion,
                                label=args.label, width=args.width, height=args.height,
                                with_agumentation=args.with_agumentation,depth=args.depth, deepscene=args.deepscene), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid_path = args.valid_path, valid=True,
                                rgb_folder_name=args.rgb_folder_name,
                                depth_folder_name=args.depth_folder_name,
                                label_folder_name=args.label_folder_name,
                                sensor_fusion=args.sensor_fusion,
                                label=args.label, width=args.width, height=args.height, depth=args.depth, deepscene=args.deepscene),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_miou = 0
    patience = args.patience
    counter = 0
    best_model_path = os.path.join(args.savedir, 'best_model.pth')
    last_checkpoint_path = os.path.join(args.savedir, 'last_checkpoint.pth.tar')
    # file csv were you want to save performance metrics for each epoch
    csv_file = os.path.join(args.savedir, 'metrics.csv')
    #write the header of the csv file
    with open(csv_file, 'w') as f:
        f.write('epoch,da_seg_acc,da_seg_iou,da_seg_miou,da_seg_acc_OFFNET,da_seg_pre_OFFNET,da_seg_recall_OFFNET,da_seg_f1_OFFNET,da_seg_iou_OFFNET,train_loss,val_loss\n')

    for epoch in range(start_epoch, args.max_epochs):
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        model.train()

        if args.sensor_fusion:
            train_loss = train_sensor_fusion(args, trainLoader, model, criteria, optimizer, epoch)
        else:
            train_loss = train(args, trainLoader, model, criteria, optimizer, epoch)
        msg = 'Epoch: [{0}/{1}]\t lr: {2}\t Train Loss {3:.4f}'.format(epoch, args.max_epochs, lr, train_loss)
        print(msg)
        # validation
        model.eval()
        if args.sensor_fusion:
            da_segment_results, da_segment_result_OFFNET, val_loss = val_sensor_fusion(valLoader, model, criteria)
        else:
            da_segment_results, da_segment_result_OFFNET, val_loss = val(valLoader, model, criteria)
        #current_miou = da_segment_results[2]  # Assuming mIOU is the third value in da_segment_results
        current_iou_tru_class = da_segment_result_OFFNET[4]
        msg = 'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})  Val_Loss({val_loss:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2], val_loss=val_loss)
        print(msg)
        msg_OFFNET = 'Driving area Segment OFFNET metrics: Acc({da_seg_acc:.3f})    Precision ({da_seg_pre:.3f})    Recall({da_seg_recall:.3f})    F1({da_seg_f1:.3f})    IoU[0]({da_seg_iou_0:.3f}) IoU[1]({da_seg_iou:.3f})'.format(
            da_seg_acc=da_segment_result_OFFNET[0], da_seg_pre=da_segment_result_OFFNET[1], da_seg_recall=da_segment_result_OFFNET[2], da_seg_f1=da_segment_result_OFFNET[3], da_seg_iou=da_segment_result_OFFNET[4], da_seg_iou_0=da_segment_result_OFFNET[5])
        print(msg_OFFNET)

        # Save the metrics to a csv file
        with open(csv_file, 'a') as f:
            f.write(f"{epoch},{da_segment_results[0]},{da_segment_results[1]},{current_iou_tru_class},{da_segment_result_OFFNET[0]},{da_segment_result_OFFNET[1]},{da_segment_result_OFFNET[2]},{da_segment_result_OFFNET[3]},{da_segment_result_OFFNET[4]},{train_loss},{val_loss}\n")



        # Save the last checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr,
            'best_miou': best_miou
        }, last_checkpoint_path)

        # Check if this is the best model
        if current_iou_tru_class > best_miou:
            best_miou = current_iou_tru_class
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with mIOU: {best_miou:.4f}")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training completed. Best mIOU: {best_miou:.4f}")
#default='pretrained//twinlitenet_encoder_freezed_up_to_b1_w_pretraining.pth'
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', required=True, help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--train_path', required=True, help='Path to the training dataset')
    parser.add_argument('--valid_path', required=True, help='Path to the validation dataset')
    parser.add_argument('--label', default="BDD100K", help='select the label type to use')
    parser.add_argument('--sensor_fusion', default=0, help='Enable sensor fusion')
    parser.add_argument('--rgb_folder_name', default="color", help='Folder name for RGB images')
    parser.add_argument('--label_folder_name', default="labels", help='Folder name for label images')
    parser.add_argument('--depth_folder_name', default="depth", help='Folder name for depth images')
    parser.add_argument('--width', type=int, default=640, help='Width of the input image')
    parser.add_argument('--height', type=int, default=360, help='Height of the input image')
    parser.add_argument('--with_agumentation', default=0, help='Enable data augmentation')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--depth', default=0, help='Enable depth images')
    parser.add_argument('--adaptive', default=0, help='Enable adaptive fusion')
    #add the encoder depth pretrained weights
    parser.add_argument('--encoder_pretrained_depth', help='Pretrained ESPNetv2 weights for depth encoder.')
    parser.add_argument('--encoder_pretrained_rgb', help='Pretrained ESPNetv2 weights for rgb encoder.')
    parser.add_argument('--freeze_encoder', default=0, help='Freeze the encoder weights')
    parser.add_argument('--deepscene', default=0, help='Enable deepscene dataset')
    
    train_net(parser.parse_args())