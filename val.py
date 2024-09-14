import torch
import torch
from loss import TotalLoss
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, netParams, val_sensor_fusion
import torch.optim.lr_scheduler
from const import *


def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''
    #print('Validation only')
    # load the model
    if args.sensor_fusion :
        print("Using RGB and Depth")
        if args.adaptive:
            model = net.TwinLiteNet_RGBD_Adaptive()
        else:
            model = net.TwinLiteNetRGBD()
    elif args.depth:
        print("Using only depth")
        model = net.TwinLiteNet()
    else:
        print("Using only RGB ")
        model = net.TwinLiteNet()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    #print('Model loaded')
    criteria = TotalLoss()
        
    #print('Validation loader ...')
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(test_path = args.test_path, test=True,\
                                rgb_folder_name=args.rgb_folder_name, \
                                depth_folder_name=args.depth_folder_name, \
                                label_folder_name=args.label_folder_name, \
                                sensor_fusion=args.sensor_fusion,\
                                label=args.label, width=args.width, height=args.height, depth=args.depth, deepscene=args.deepscene),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #print('Validation loader ready')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    #model.load_state_dict(torch.load(args.weight))
    

    checkpoint = torch.load( args.weight) # change the model
    checkpoint2 = {}
    for key in checkpoint:
        if key.startswith("module."):
            checkpoint2[key] = checkpoint[key]
        else:
            checkpoint2["module."+key] = checkpoint[key]
    #save pretrained model without lane detection
    
    model.load_state_dict(checkpoint2)
    model.eval()
    
    
    if args.sensor_fusion:
        example_rgb = torch.rand(1, 3, 360, 640).cuda()
        example_depth = torch.rand(1, 3, 360, 640).cuda()
        model = torch.jit.trace(model, (example_rgb, example_depth))
        da_segment_results, da_segment_result_OFFNET, val_loss = val_sensor_fusion(valLoader, model, criteria)
    else:
        example = torch.rand(1, 3, 360, 640).cuda()
        model = torch.jit.trace(model, example)
        da_segment_results, da_segment_result_OFFNET, val_loss = val(valLoader, model, criteria)
        
    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})  Val_Loss({val_loss:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2], val_loss=val_loss)
    print(msg)
    msg_OFFNET = 'Driving area Segment OFFNET metrics: Acc({da_seg_acc:.3f})    Precision ({da_seg_pre:.3f})    Recall({da_seg_recall:.3f})    F1({da_seg_f1:.3f})   IOU[0]({da_seg_iou_0}) IoU[1]({da_seg_iou:.3f})'.format(
                            da_seg_acc=da_segment_result_OFFNET[0],da_seg_pre=da_segment_result_OFFNET[1],da_seg_recall=da_segment_result_OFFNET[2],da_seg_f1=da_segment_result_OFFNET[3],da_seg_iou=da_segment_result_OFFNET[4], da_seg_iou_0=da_segment_result_OFFNET[5])
    print(msg_OFFNET)
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--weight', default="pretrained/best1.pth")
    parser.add_argument('--num_workers', type=int, default=1, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--test_path', required=True, help='Path to the test dataset')
    parser.add_argument('--label', default="BDD100K", help='select the label type to use')
    parser.add_argument('--sensor_fusion', default=0, help='Enable sensor fusion')
    parser.add_argument('--rgb_folder_name', default="color", help='Folder name for RGB images')
    parser.add_argument('--label_folder_name', default="labels", help='Folder name for label images')
    parser.add_argument('--depth_folder_name', default="depth", help='Folder name for depth images')
    parser.add_argument('--width', type=int, default=640, help='Width of the input image')
    parser.add_argument('--height', type=int, default=360, help='Height of the input image')
    parser.add_argument('--depth', default=0, help='Enable depth estimation')
    parser.add_argument('--adaptive', default=0, help='Enable adaptive fusion')
    parser.add_argument('--deepscene', default=0, help='Enable deepscene dataset')
    
    validation(parser.parse_args())
