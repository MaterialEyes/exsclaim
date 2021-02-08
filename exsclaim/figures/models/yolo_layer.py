import torch
import torch.nn as nn
import numpy as np
from .network import resnet152
import cv2
import warnings

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)
class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8] # fixed
        self.anchors = config_model['ANCHORS']
        self.n_anchors = len(self.anchors)
#         self.anch_mask = config_model['ANCH_MASK'][layer_no]
#         self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.ignore_thre = ignore_thre
        # self.l2_loss = nn.MSELoss(size_average=False)
        # self.bce_loss = nn.BCELoss(size_average=False)
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')


        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = self.all_anchors_grid
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * 5,
                              kernel_size=1, stride=1, padding=0)
        self.classifier_model = resnet152()

    def forward(self, xin, compound_labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # Suppresses incorrect UserWarning about a non-writeable Numpy array
        # PR with fix accepted shortly after torch 1.7.1 release
        # https://github.com/pytorch/pytorch/pull/47271
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # calculate pred - xywh obj cls
            x_shift = dtype(np.broadcast_to(
                np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if compound_labels is None:  # not training
            pred[..., :4] *= self.stride
            return pred.reshape(batchsize, -1, n_ch).data

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4).type(dtype)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)

        labels, imgs = compound_labels
        imgs = imgs.data.cpu().numpy()
        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            img = imgs[b].transpose((1,2,0))[:,:,::-1]
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, pred_best_iou_index = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            not_obj_mask = (pred_best_iou<0.3)
            
            
            is_obj_mask = (pred_best_iou>0.7)
            pred_best_iou_index = pred_best_iou_index.view(pred[b].shape[:3])
            
            obj_mask[b] = (not_obj_mask+is_obj_mask).type(dtype)
            
            # encourage bbox with over 0.7 IOU
            rp_index = np.nonzero(is_obj_mask)
            for rp_i in range(rp_index.size()[0]):
                rp_anchor, rp_i, rp_j = rp_index[rp_i]
                truth_box_index = int(pred_best_iou_index[rp_anchor, rp_i, rp_j])
                target[b,rp_anchor, rp_i, rp_j,4] = 1
                
                # target label for the bbox
                reference_label = int(labels[b, truth_box_index, 0])
                
                self.classifier_model.eval()

                pred_x,pred_y,pred_w,pred_h = pred[b,rp_anchor, rp_i, rp_j,:4]*self.stride
                x1 = int(min(max(pred_x-pred_w/2,0),img.shape[0]-1))
                y1 = int(min(max(pred_y-pred_h/2,0),img.shape[0]-1))
                x2 = int(min(max(pred_x+pred_w/2,0),img.shape[0]-1))
                y2 = int(min(max(pred_y+pred_h/2,0),img.shape[0]-1))
#                 print(x1,y1,x2,y2)
                if (x1+2 < x2) and (y1+2<y2):
                    patch = np.uint8(255*img[y1:y2,x1:x2])
#                     print("patch size is", patch.shape)
                    patch, _ = preprocess(patch, 28, jitter=0)
                    patch = np.transpose(patch/255.0, (2, 0, 1))
                    patch = torch.from_numpy(patch).unsqueeze(0).type(dtype)
                    pred_label = int(self.classifier_model(patch).argmax(dim=1).data.cpu().numpy()[0])
                
                    if pred_label == reference_label:
                        target[b,rp_anchor, rp_i, rp_j,4] = 1
                    else:
#                         print("%d/%d, pred=%d, gt=%d"%(rp_i,int(rp_index.size()[0]),pred_label,reference_label))
                        target[b,rp_anchor, rp_i, rp_j,4] = 0
                else:
                    target[b,rp_anchor, rp_i, rp_j,4] = 0
                
                
            


            for ti in range(best_n_all.shape[0]):
#                 if best_n_mask[ti] == 1:
                i, j = truth_i[ti], truth_j[ti]
                a = best_n_all[ti]
                obj_mask[b, a, j, i] = 1
                tgt_mask[b, a, j, i, :] = 1
                target[b, a, j, i, 0] = truth_x_all[b, ti] - i
                target[b, a, j, i, 1] = truth_y_all[b, ti] - j
                target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] /w_anchors[b,a,j,i]  + 1e-16)
                target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] /h_anchors[b,a,j,i]  + 1e-16)
                target[b, a, j, i, 4] = 1
                tgt_scale[b, a, j, i, :] = torch.sqrt(
                    2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
                
                


        # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,
                             reduction='sum')  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = 0#self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


class YOLOimgLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOimgLayer, self).__init__()
        strides = [32, 16, 8] # fixed
        self.anchors = config_model['ANCHORS']
        self.n_anchors = len(self.anchors)
#         self.anch_mask = config_model['ANCH_MASK'][layer_no]
#         self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = self.all_anchors_grid
#         self.masked_anchors = [self.all_anchors_grid[i]
#                                for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)

    def forward(self, xin, all_labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)
        labels, prior_labels = all_labels

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()
#         print(output.size())

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])
#         output[..., np.r_[2:n_ch]] = torch.sigmoid(
#             output[..., np.r_[2:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        pred[..., :2] -= 0.5
        pred[..., 0] *= w_anchors
        pred[..., 1] *= h_anchors
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors
        
        prior_labels = prior_labels.cpu().data
        nprior_label = (prior_labels.sum(dim=2) > 0).sum(dim=1)
        truth_x_all_sub = prior_labels[:, :, 1] * fsize
        truth_y_all_sub = prior_labels[:, :, 2] * fsize
        truth_i_all_sub = truth_x_all_sub.to(torch.int16).numpy()
        truth_j_all_sub = truth_y_all_sub.to(torch.int16).numpy()

    #         pred_mask = dtype(np.zeros((batchsize, self.n_anchors, fsize, fsize, n_ch)))
    #         for b in range(batchsize):
    #             for ti in range(nprior_label[b]):
    #                 i,j = truth_i_all_sub[b,ti], truth_j_all_sub[b,ti]
    #                 best_anchor = torch.argmax(pred[b,:,j,i,4])
    #                 i_best = min(max(int(pred[b,best_anchor,j,i,0].to(torch.int16).cpu().numpy()),0),fsize-1)
    #                 j_best = min(max(int(pred[b,best_anchor,j,i,1].to(torch.int16).cpu().numpy()),0),fsize-1)
    # #                 if labels is None:
    # #                     pass
    # #                 else:
    # #                     pred_mask[b,:,j,i,:] = 1 
    #                 pred_mask[b,:,j,i,:] = 1 
    #                 pred_mask[b,best_anchor,j_best,i_best,:] = 1
    #         pred *= pred_mask
    

        if labels is None:  # not training
            pred[..., :4] *= self.stride
            return pred.data
#             return pred.view(batchsize, -1, n_ch).data

        pred = pred[..., :4].data

        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)
        in_grid_distance = torch.zeros(batchsize, 80 ,2).type(dtype)

#         tgt_mask = torch.zeros(batchsize, self.n_anchors,
#                                fsize, fsize, 4 + self.n_classes).type(dtype)
# #         obj_mask = torch.ones(batchsize, self.n_anchors,
# #                               fsize, fsize).type(dtype)
#         obj_mask = torch.zeros(batchsize, self.n_anchors,
#                               fsize, fsize).type(dtype)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
#         assert nprior_label == nlabel

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
#         truth_i_all = truth_x_all.to(torch.int16).numpy()
#         truth_j_all = truth_y_all.to(torch.int16).numpy()

#         pred_areas = torch.zeros(batchsize).type(dtype)
#         target_areas = torch.zeros(batchsize).type(dtype)
        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all_sub[b, :n]
            truth_j = truth_j_all_sub[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box, (self.ref_anchors).type(dtype))
            best_n_all = torch.argmax(anchor_ious_all, dim=1)

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

#             pred_ious = bboxes_iou(
#                 pred[b].view(-1, 4), truth_box, xyxy=False)
#             pred_best_iou, _ = pred_ious.max(dim=1)
#             pred_best_iou = (pred_best_iou > self.ignore_thre)
#             pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
#             # set mask to zero (ignore) if pred matches truth
#             obj_mask[b] = 1- pred_best_iou

#             if sum(best_n_mask) == 0:
#                 continue
 
            for ti in range(n):
                i, j = truth_i[ti], truth_j[ti]
                
                # find box with iou over 0.7 and under 0.3 (achor point)
                current_truth_box = truth_box[ti:ti+1]
                current_pred_boxes = pred[b,:,j,i,:4]
                pred_ious = bboxes_iou(current_truth_box, current_pred_boxes, xyxy=False)
                good_anchor_index = torch.nonzero((pred_ious>0.7)[0]).cpu().numpy()
                bad_anchor_index = torch.nonzero((pred_ious<0.3)[0]).cpu().numpy()
#                 print(len(good_anchor_index),len(bad_anchor_index),good_anchor_index, bad_anchor_index)
                for good_i in range(len(good_anchor_index)):
                    a = good_anchor_index[good_i]
                    tgt_mask[b,a,j,i,:] = 1
                    target[b, a, j, i, 0] = torch.clamp((truth_x_all[b, ti]-i)/torch.Tensor(self.masked_anchors)[a, 0]+0.5,0,1)
                    target[b, a, j, i, 1] = torch.clamp((truth_y_all[b, ti]-j)/torch.Tensor(self.masked_anchors)[a, 1]+0.5,0,1)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
                    
                    i_best = min(max(int(pred[b,a,j,i,0].cpu().numpy()),0),fsize-1)
                    j_best = min(max(int(pred[b,a,j,i,1].cpu().numpy()),0),fsize-1)
                    current_pred_boxes_2 = pred[b,:,j_best,i_best,:4]
                    pred_ious_2 = bboxes_iou(current_truth_box, current_pred_boxes_2, xyxy=False)
                    good_anchor_index_2 = torch.nonzero((pred_ious_2>0.7)[0]).cpu().numpy()
                    bad_anchor_index_2 = torch.nonzero((pred_ious_2<0.3)[0]).cpu().numpy()
    #                 print(len(good_anchor_index),len(bad_anchor_index),good_anchor_index, bad_anchor_index)
                    for good_i_2 in range(len(good_anchor_index_2)):
                        a = good_anchor_index_2[good_i_2]
                        tgt_mask[b,a,j_best,i_best,:] = 1
                        target[b, a, j_best, i_best, 0] = torch.clamp((truth_x_all[b, ti]-i_best)/torch.Tensor(self.masked_anchors)[a, 0]+0.5,0,1)
                        target[b, a, j_best, i_best, 1] = torch.clamp((truth_y_all[b, ti]-j_best)/torch.Tensor(self.masked_anchors)[a, 1]+0.5,0,1)
                        target[b, a, j_best, i_best, 2] = torch.log(
                            truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 0] + 1e-16)
                        target[b, a, j_best, i_best, 3] = torch.log(
                            truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 1] + 1e-16)
                        target[b, a, j_best, i_best, 4] = 1
                        target[b, a, j_best, i_best, 5 + labels[b, ti,
                                                      0].to(torch.int16).numpy()] = 1
                        tgt_scale[b, a, j_best, i_best, :] = torch.sqrt(
                            2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

                    for bad_i_2 in range(len(bad_anchor_index_2)):
                        a = bad_anchor_index_2[bad_i_2]
                        tgt_mask[b,a,j_best,i_best,4:] = 1
                    
                    
                    
                for bad_i in range(len(bad_anchor_index)):
                    a = bad_anchor_index[bad_i]
                    tgt_mask[b,a,j,i,4:] = 1
                
                # best anchor box
                a = best_n_all[ti]
                tgt_mask[b,a,j,i,:] = 1
                target[b, a, j, i, 0] = torch.clamp((truth_x_all[b, ti]-i)/torch.Tensor(self.masked_anchors)[a, 0]+0.5,0,1)
                target[b, a, j, i, 1] = torch.clamp((truth_y_all[b, ti]-j)/torch.Tensor(self.masked_anchors)[a, 1]+0.5,0,1)
                target[b, a, j, i, 2] = torch.log(
                    truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 0] + 1e-16)
                target[b, a, j, i, 3] = torch.log(
                    truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 1] + 1e-16)
                target[b, a, j, i, 4] = 1
                target[b, a, j, i, 5 + labels[b, ti,
                                              0].to(torch.int16).numpy()] = 1
                tgt_scale[b, a, j, i, :] = torch.sqrt(
                    2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
                
                i_best = min(max(int(pred[b,a,j,i,0].cpu().numpy()),0),fsize-1)
                j_best = min(max(int(pred[b,a,j,i,1].cpu().numpy()),0),fsize-1)
                
                 # find box with iou over 0.7 and under 0.3 (predict center)
                current_truth_box = truth_box[ti:ti+1]
                current_pred_boxes = pred[b,:,j_best,i_best,:4]
                pred_ious = bboxes_iou(current_truth_box, current_pred_boxes, xyxy=False)
                good_anchor_index = torch.nonzero((pred_ious>0.7)[0]).cpu().numpy()
                bad_anchor_index = torch.nonzero((pred_ious<0.3)[0]).cpu().numpy()
#                 print(len(good_anchor_index),len(bad_anchor_index),good_anchor_index, bad_anchor_index)
                for good_i in range(len(good_anchor_index)):
                    a = good_anchor_index[good_i]
                    tgt_mask[b,a,j_best,i_best,:] = 1
                    target[b, a, j_best, i_best, 0] = torch.clamp((truth_x_all[b, ti]-i_best)/torch.Tensor(self.masked_anchors)[a, 0]+0.5,0,1)
                    target[b, a, j_best, i_best, 1] = torch.clamp((truth_y_all[b, ti]-j_best)/torch.Tensor(self.masked_anchors)[a, 1]+0.5,0,1)
                    target[b, a, j_best, i_best, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 0] + 1e-16)
                    target[b, a, j_best, i_best, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 1] + 1e-16)
                    target[b, a, j_best, i_best, 4] = 1
                    target[b, a, j_best, i_best, 5 + labels[b, ti,
                                                  0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j_best, i_best, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
                    
                for bad_i in range(len(bad_anchor_index)):
                    a = bad_anchor_index[bad_i]
                    tgt_mask[b,a,j_best,i_best,4:] = 1
                
                
                a = best_n_all[ti]
                tgt_mask[b,a,j_best,i_best,:] = 1
                target[b, a, j_best, i_best, 0] = torch.clamp((truth_x_all[b, ti]-i_best)/torch.Tensor(self.masked_anchors)[a, 0]+0.5,0,1)
                target[b, a, j_best, i_best, 1] = torch.clamp((truth_y_all[b, ti]-j_best)/torch.Tensor(self.masked_anchors)[a, 1]+0.5,0,1)
                target[b, a, j_best, i_best, 2] = torch.log(
                    truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 0] + 1e-16)
                target[b, a, j_best, i_best, 3] = torch.log(
                    truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[a, 1] + 1e-16)
                target[b, a, j_best, i_best, 4] = 1
                target[b, a, j_best, i_best, 5 + labels[b, ti,
                                              0].to(torch.int16).numpy()] = 1
                tgt_scale[b, a, j_best, i_best, :] = torch.sqrt(
                    2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
                
#                 in_grid_distance[b,ti,0] += target[b, a, j_best, i_best, 0]
#                 in_grid_distance[b,ti,1] += target[b, a, j_best, i_best, 1]
                

        # loss calculation
    
        output *= tgt_mask
        target *= tgt_mask
        target_in_grid_distance = torch.zeros(batchsize, 80 ,2).type(dtype)

#         output[..., 4] *= obj_mask
#         output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

#         target[..., 4] *= obj_mask
#         target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,
                             reduction='sum')  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)
        loss_in_grid = self.bce_loss(in_grid_distance, target_in_grid_distance)
        
#         target_areas = torch.ones(batchsize).type(dtype)
#         loss_area = 0.1*self.l2_loss(pred_areas,target_areas)

        loss = loss_xy + loss_wh + loss_obj + loss_cls + 0.01*loss_in_grid

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_in_grid, loss_l2

