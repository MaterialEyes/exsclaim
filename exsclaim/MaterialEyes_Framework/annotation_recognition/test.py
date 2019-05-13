import argparse
import yaml

import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights


import glob
import os
from dataset.figsepdataset import *


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the checkpoint file')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='path to weights file')
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--image_extend', type=str,default="jpg")
    parser.add_argument('--groundtruth', action="store_true")
    parser.add_argument('--gt_data_dir', type=str)
    parser.add_argument('--gt_anno_dir', type=str)
    parser.add_argument('--max_img', type=int,default=2000)
    parser.add_argument('--background', action='store_true',
                        default=False, help='background(no-display mode. save "./output.png")')
    parser.add_argument('--detect_thresh', type=float,
                        default=None, help='confidence threshold')
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    
    if args.groundtruth:
        args.cfg = 'config/yolov3_eval.cfg'
        with open(args.cfg, 'r') as f:
            cfg = yaml.load(f)

        imgsize = cfg['TEST']['IMGSIZE']
        model = YOLOv3(cfg['MODEL'])

        confthre = cfg['TEST']['CONFTHRE'] 
        nmsthre = cfg['TEST']['NMSTHRE']
        batch_size = cfg['TRAIN']['BATCHSIZE']
        
        if args.detect_thresh:
            confthre = args.detect_thresh
            
        dataset = FigSepDataset(model_type=cfg['MODEL']['TYPE'],
                              data_dir=args.gt_data_dir,
                              anno_dir = args.gt_anno_dir,
                              img_size=cfg['TRAIN']['IMGSIZE'],
                              augmentation=cfg['AUGMENTATION'],
                              debug=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        dataiterator = iter(dataloader)
        
        cuda = torch.cuda.is_available()
        
        if cuda:
            print("using cuda") 
            model = model.cuda()
        
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        total_loss = 0
        for i in range(args.max_img):
            try:
                imgs, targets, _, _ = next(dataiterator)
            except StopIteration:
                print("{} images has been assessed".format(i+1))
                break
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
            loss = model(imgs, targets)
            total_loss += loss.data.cpu().numpy()
        print("loss is {}".format(total_loss/(i+1)))
        return
    else:
        pass
    
    
    
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    imgsize = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])

    confthre = cfg['TEST']['CONFTHRE'] 
    nmsthre = cfg['TEST']['NMSTHRE']
#     confthre = 0
#     nmsthre = 1

    if args.detect_thresh:
        confthre = args.detect_thresh

    if args.image_dir:
        img_names = glob.glob(os.path.join(args.image_dir,"*."+args.image_extend))
        print("num of images:{}".format(len(img_names)))
    else:
        img_names = [args.image]
        print("num of images:{}".format(len(img_names)))
    
    
    
    
    
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    for sample_img in img_names:
        args.image = sample_img
    
        img = cv2.imread(args.image)
        img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img, imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if args.gpu >= 0:
            model.cuda(args.gpu)
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))

        if args.weights_path:
            print("loading yolo weights %s" % (args.weights_path))
            parse_yolo_weights(model, args.weights_path)
        else:
            print("loading checkpoint %s" % (args.ckpt))
            model.load_state_dict(torch.load(args.ckpt)["model_state_dict"])

        model.eval()

        with torch.no_grad():
            outputs = model(img)
            print("outputs:",outputs.size())
            outputs = postprocess(outputs, 80, confthre, nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            continue

        coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

        bboxes = list()
        classes = list()
        colors = list()
        confidences = []

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:

            cls_id = coco_class_ids[int(cls_pred)]
            print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            print('\t+ Label: %s, Conf: %.5f' %
                  (coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2label([y1, x1, y2, x2], info_img)
            bboxes.append(box)
            classes.append(cls_id)
            colors.append(coco_class_colors[int(cls_pred)])
            confidences.append("%.3f"%(cls_conf.item()))

        
        from PIL import Image
        test_img = Image.open(args.image).convert("RGB")
        sub_figures = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            h,w = test_img.size
            y0 = int(max(box[0].data.cpu().numpy(),0))
            x0 = int(max(box[1].data.cpu().numpy(),0))
            y1 = int(min(box[2].data.cpu().numpy(),w-1))
            x1 = int(min(box[3].data.cpu().numpy(),h-1))
#             print(x0,y0,x1,y1,type(x0))
            sub_figure = test_img.crop((x0,y0,x1,y1))
            sub_figures.append(sub_figure)
#         if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
#             os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
        for i in range(len(sub_figures)):
            sub_figures[i].save(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]+"----{}_output.png".format(i+1)))
#             sub_figures[i].save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"{}_output.png".format(i+1)))
        
        
#         from PIL import Image, ImageDraw, ImageFont
#         test_img = Image.open(args.image).convert("RGB")
#         draw = ImageDraw.Draw(test_img)
#         font_type = "arial.ttf"
#         font = ImageFont.truetype(font_type, 10)

#         for i in range(len(bboxes)):
#             box = bboxes[i]
#             h,w = test_img.size
#             y0 = max(box[0].data.cpu().numpy(),0)
#             x0 = max(box[1].data.cpu().numpy(),0)
#             y1 = min(box[2].data.cpu().numpy(),w-1)
#             x1 = min(box[3].data.cpu().numpy(),h-1)
#             draw.rectangle([x0,y0,x1,y1],outline="blue")#tuple(colors[i]))
#             label = coco_class_names[classes[i]]
#             conf = confidences[i]
#             text = label+":"+conf
#             text_width, text_height = font.getsize(text)
#             draw.text((x0,y0),text,fill="red",font=font)
#         del draw
#         if not os.path.exists(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0])):
#             os.makedirs(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]))
#         test_img.save(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]+"output.png"))
#         test_img.save(os.path.join(os.path.join(args.result_dir,sample_img.split("/")[-1].split(".")[0]),"output.png"))
    
    
#     print("bbbox=",bboxes)
#     print("classes=",classes)
#     print("colors=",colors)
#     print(tuple(colors[0]))

#     if args.background:
#         import matplotlib
#         matplotlib.use('Agg')

#     from utils.vis_bbox import vis_bbox
#     import matplotlib.pyplot as plt

#     vis_bbox(
#         img_raw, bboxes, label=classes, label_names=coco_class_names,
#         instance_colors=colors, linewidth=2)
#     plt.show()

#     if args.background:
#         plt.savefig('output.png')


if __name__ == '__main__':
    main()
