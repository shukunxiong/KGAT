import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_bbox import dist2bbox, make_anchors
import numpy as np
import cv2  # 用于计算灰度均值
import scipy.stats as stats


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors       = xy_centers.shape[0]
    bs, n_boxes, _  = gt_bboxes.shape
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb          = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            bbox_deltas[b]  = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        lt, rb      = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # b, n_max_boxes, 8400 -> b, 8400
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  
        # b, n_max_boxes, 8400
        mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  
        # b, 8400
        max_overlaps_idx    = overlaps.argmax(1)  
        # b, 8400, n_max_boxes
        is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)  
        # b, n_max_boxes, 8400
        is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  
        # b, n_max_boxes, 8400
        mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) 
        fg_mask             = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk           = topk
        self.num_classes    = num_classes
        self.bg_idx         = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        # roll_out_thr为64
        self.roll_out_thr   = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """
        self.bs             = pd_scores.size(0)
        self.n_max_boxes    = gt_bboxes.size(1)
        self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False
    
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # b, max_num_obj, 8400
        # mask_pos      
        # align_metric  
        # overlaps      
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        # target_gt_idx     b, 8400     
        # fg_mask           b, 8400     
        # mask_pos          b, max_num_obj, 8400    
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric        *= mask_pos
        # b, max_num_obj
        pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 
        # b, max_num_obj
        pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores       = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores bs, num_total_anchors, num_classes
        # pd_bboxes bs, num_total_anchors, 4
        # gt_labels bs, n_max_boxes, 1
        # gt_bboxes bs, n_max_boxes, 4
        # 
        # align_metric, overlaps    bs, max_num_obj, 8400
        align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        
        # get in_gts mask           b, max_num_obj, 8400

        mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask      b, max_num_obj, 8400
        mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, b, max_num_obj, h*w
        mask_pos                = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # bs, max_num_obj, 8400
                bbox_scores     = pd_scores[ind_0, :, ind_2]  
                # bs, max_num_obj, 8400
                overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # 2, b, max_num_obj
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       
            # b, max_num_obj  
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  
            ind[1] = gt_labels.long().squeeze(-1) 
            # b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]  

            # bs, max_num_obj, 8400
            overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
            align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics     : (b, max_num_obj, h*w).
            topk_mask   : (b, max_num_obj, topk) or None
        """
        # 8400
        num_anchors             = metrics.shape[-1] 
        # b, max_num_obj, topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, topk
        topk_idxs[~topk_mask] = 0
        # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels       : (b, max_num_obj, 1)
            gt_bboxes       : (b, max_num_obj, 4)
            target_gt_idx   : (b, h*w)
            fg_mask         : (b, h*w)
        """

        batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # b, h*w   
        target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
        # b, h*w   
        target_labels   = gt_labels.long().flatten()[target_gt_idx]
        # b, h*w, 4 
        target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels.clamp(0)
        target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum


        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

# Criterion class for computing training losses
class Loss:
    def __init__(self, model): 
        self.bce    = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc     = model.num_classes  # number of classes
        self.no     = model.no
        self.reg_max = model.reg_max
        
        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss  = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj       = torch.arange(model.reg_max, dtype=torch.float)
    

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            i           = targets[:, 0]  
            _, counts   = i.unique(return_counts=True)
            out         = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  
            # DFL的解码
            pred_dist   = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        device  = preds[1].device
        loss    = torch.zeros(3, device=device)  
        feats   = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]
        imgsz       = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)


        targets                 = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # bs, max_boxes_num, 5
        targets                 = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes    = targets.split((1, 4), 2)  # cls, xyxy
        # bs, max_boxes_num
        mask_gt                 = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # bs, 8400, 4
        pred_bboxes             = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes       /= stride_tensor
        target_scores_sum   = max(target_scores.sum(), 1)

        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum() # loss(box, cls, dfl) # * batch_size

class Loss_Knowledge:
    def __init__(self, model,reweight_dir,exponent): 
        self.bce    = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc     = model.num_classes  # number of classes
        self.no     = model.no
        self.reg_max = model.reg_max
        self.exponent = exponent
        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss  = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj       = torch.arange(model.reg_max, dtype=torch.float)
        self.hard_tensor  = self.hard_find(reweight_dir)
        
    def hard_find(self,reweight_dir):
        tensor = torch.load(reweight_dir)
        
        return tensor

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # Obtain the image index
            i           = targets[:, 0]  
            _, counts   = i.unique(return_counts=True)
            out         = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  
            pred_dist   = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def compute_weighted_difficulty(self, gt_label):
        gt_label = torch.tensor(gt_label)
        
        unique_classes, class_counts = torch.unique(gt_label, return_counts=True)
        unique_classes = unique_classes.int()
        n = len(unique_classes)
        
        # Total number of samples
        total_count = gt_label.size(0)
        if total_count ==0:
            return 1
        
        self.hard_tensor = self.hard_tensor.to(unique_classes.device)
        submatrix = self.hard_tensor[unique_classes[:, None], unique_classes[None, :]] 
        submatrix = torch.abs(submatrix)
        matrix_sum = submatrix.sum()/(n**2-n)
        matrix_sum = 0.95+0.1*matrix_sum 
        
        return matrix_sum

    def __call__(self, images,preds, batch):
        device  = preds[1].device
        # Losses for the three components: bounding box (box), classification (cls), and distribution focal loss (dfl)
        loss    = torch.zeros(3, device=device)  
        feats   = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]
        imgsz       = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)


        targets                 = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # bs, max_boxes_num, 5
        targets                 = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes    = targets.split((1, 4), 2)  # cls, xyxy
        # bs, max_boxes_num
        mask_gt                 = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # bs, 8400, 4
        pred_bboxes             = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes       /= stride_tensor
        target_scores_sum   = max(target_scores.sum(), 1)

        # Scaling ratio of the feature map size relative to the original image
        rate = 8
        pre_gray_values, gt_gray_values=compute_grayscale_means(target_bboxes, target_scores, fg_mask, gt_labels, gt_bboxes, mask_gt, images, rate, topk=5, threshold=0.1)
        spearman_corr = compute_spearman(pre_gray_values, gt_gray_values)
        
        #-------Thermal Radiation Relation-Guided Optimization-----------
        beta = 0.5
        spearman_corr = torch.tensor(spearman_corr,device=target_bboxes.device)
        spearman_corr = -torch.log(beta*spearman_corr+1)+1
        #--------Knowledge Reliability-Aware Optimization----------------

        mask_hard = mask_gt.bool().squeeze() 
        gt_labels = gt_labels[mask_hard]
        hard_weight = self.compute_weighted_difficulty(gt_labels)
        
        reweight = spearman_corr * hard_weight
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5 * reweight  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5 * reweight # dfl gain
        return loss.sum() # loss(box, cls, dfl) # * batch_size

def compute_spearman(pre_gray_values, gt_gray_values):
    # Step 1: Filter out values where both are greater than 0
    mask =( pre_gray_values + gt_gray_values )> 0
    pre_gray_values_filtered = pre_gray_values[mask]
    gt_gray_values_filtered = gt_gray_values[mask]

    # Step 2: Check for NaN or inf values
    if torch.any(torch.isnan(pre_gray_values_filtered)) or torch.any(torch.isinf(pre_gray_values_filtered)) or \
       torch.any(torch.isnan(gt_gray_values_filtered)) or torch.any(torch.isinf(gt_gray_values_filtered)):
        raise ValueError("One of the tensors contains NaN or inf values.")

    # Step 3: Check conditions for r = 0
    # If there's only one non-zero element in either tensor, set r = 0
    if (torch.sum(pre_gray_values_filtered > 0) == 1) or (torch.sum(gt_gray_values_filtered > 0) == 1):
        return 0
    
    if torch.all(pre_gray_values_filtered == pre_gray_values_filtered[0]) or torch.all(gt_gray_values_filtered == gt_gray_values_filtered[0]):
        # If all elements are the same in any of the filtered tensors, set r = 0
        return 0

    # Step 4: Calculate Spearman rank correlation if conditions are met
    spearman_corr, _ = stats.spearmanr(pre_gray_values_filtered.cpu().numpy(), gt_gray_values_filtered.cpu().numpy())
    spearman_corr_tensor = torch.tensor(spearman_corr)
    return spearman_corr_tensor  

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def compute_grayscale_means(target_bboxes, target_scores, fg_mask, gt_labels, gt_bboxes, mask_gt, Images_adjust, rate, topk=5, threshold=0.1):
    batch_size = target_bboxes.shape[0]
    predicted_grayscale_means = []
    labeled_grayscale_means = []
    mask_gt = mask_gt.bool().squeeze() 
    
    for i in range(batch_size):
        # 获取该批次的数据
        fg_mask_batch = fg_mask[i]
        target_bboxes_batch = target_bboxes[i]
        target_scores_batch = target_scores[i]
        gt_labels_batch = gt_labels[i]
        gt_bboxes_batch = gt_bboxes[i]
        mask_gt_batch = mask_gt[i]
        image_new = Images_adjust[i]
        image = 0.2989 * image_new[0] + 0.5870 * image_new[1] + 0.1140 * image_new[2]

        # 过滤无效框
        target_bboxes_batch = target_bboxes_batch[fg_mask_batch]
        target_scores_batch = target_scores_batch[fg_mask_batch]

        # 过滤ground truth框
        gt_bboxes_batch = gt_bboxes_batch[mask_gt_batch]
        gt_labels_batch = gt_labels_batch[mask_gt_batch]

        # 对边界框进行缩放和调整，确保框不超过图像边界且是整数
        target_bboxes_batch = torch.clamp(target_bboxes_batch * rate, min=0, max=image.size(1))
        gt_bboxes_batch = torch.clamp(gt_bboxes_batch , min=0, max=image.size(1))

        # 将target_scores转换为softmax输出
        target_scores_softmax = F.softmax(target_scores_batch, dim=-1)
        
        # --------防止什么都没有预测到，主要是针对引入对抗样本后的情况
        if target_scores_softmax.numel() == 0:  # 判断张量是否为空
            continue  # 继续循环

        # 1. 计算预测框的灰度均值
        predicted_class_grayscale_means = torch.zeros(target_scores_softmax.shape[1], device=image.device)

        for class_idx in range(target_scores_softmax.shape[1]):
            # 对每个类别进行处理，选择softmax值大于阈值τ的边界框
            scores_for_class = target_scores_softmax[:, class_idx]
            topk_indices = torch.topk(scores_for_class, min(topk,scores_for_class.size(0))).indices
            selected_boxes = target_bboxes_batch[topk_indices]
            selected_scores = scores_for_class[topk_indices]
            selected_mask = selected_scores > threshold
            selected_boxes = selected_boxes[selected_mask]
            selected_scores = selected_scores[selected_mask]

            # 如果有选中的边界框，进行灰度均值计算
            if selected_boxes.size(0) > 0:
                grayscale_mean = 0
                k = 0
                for box in selected_boxes:
                    if box.numel() == 0:
                        continue
                    # 提取对应的区域
                    image_width = image.size(1)  # 图像的宽度
                    image_height = image.size(0)  # 图像的高度
                    x1, y1, x2, y2 = box.int().tolist()
                    x1 = max(0, min(x1, image_width - 1))
                    y1 = max(0, min(y1, image_height - 1))
                    x2 = max(0, min(x2, image_width - 1))
                    y2 = max(0, min(y2, image_height - 1))
                    region = image[y1:y2, x1:x2]  # 获取区域
                    if region.size == 0:
                        continue
                    # 计算灰度均值
                    grayscale_mean += torch.mean(region)
                    k += 1
                if k>0:
                    predicted_class_grayscale_means[class_idx] = grayscale_mean/k
                else:
                    predicted_class_grayscale_means[class_idx] = 0
            else:
                # 如果没有框，则灰度均值为0
                predicted_class_grayscale_means[class_idx] = 0

        # 2. 计算真实框的灰度均值
        labeled_class_grayscale_means = torch.zeros(target_scores_softmax.shape[1], device=image.device)

        for class_idx in range(target_scores_softmax.shape[1]):  # 处理所有真实标签
            # 只选择对应类别的gt框
            gt_class_boxes = gt_bboxes_batch[gt_labels_batch.squeeze() == class_idx]
            grayscale_mean = 0
            k = 0
            if gt_class_boxes.size(0) > 0:
                for box in gt_class_boxes:
                    # 提取对应的区域
                    if box.dim() >= 2:
                        box = box.squeeze(0)
                    if not isinstance(box.int().tolist(), list) or len(box.int().tolist()) != 4:
                        continue  # 无效 box，跳过
                    x1, y1, x2, y2 = box.int().tolist()
                    image_width = image.size(1)  # 图像的宽度
                    image_height = image.size(0)  # 图像的高度
                    x1, y1, x2, y2 = box.int().tolist()
                    x1 = max(0, min(x1, image_width - 1))
                    y1 = max(0, min(y1, image_height - 1))
                    x2 = max(0, min(x2, image_width - 1))
                    y2 = max(0, min(y2, image_height - 1))
                    region = image[y1:y2, x1:x2]  # 获取区域
                    if region.size == 0:
                        continue
                    # 计算灰度均值
                    grayscale_mean += torch.mean(region)
                    k += 1
                if k>0:
                    labeled_class_grayscale_means[class_idx] = grayscale_mean/k
                else:
                    labeled_class_grayscale_means[class_idx] = 0
            else:
                # 如果没有框，则灰度均值为0
                labeled_class_grayscale_means[class_idx] = 0

        # 计算背景的灰度均值
        remaining_mask = torch.ones_like(image).bool()
        for box in selected_boxes:
            x1, y1, x2, y2 = box.int().tolist()
            image_width = image.size(1)  # 图像的宽度
            image_height = image.size(0)  # 图像的高度
            x1, y1, x2, y2 = box.int().tolist()
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))
            remaining_mask[y1:y2, x1:x2] = False

        background_region = image[remaining_mask]
        if background_region.size == 0:
            background_grayscale_mean = 0
        else:
            background_grayscale_mean = torch.mean(background_region)

        # 将背景灰度值加到类别灰度值的末尾
        predicted_class_grayscale_means = torch.cat([predicted_class_grayscale_means, torch.tensor([background_grayscale_mean], device=image.device)])
        labeled_class_grayscale_means = torch.cat([labeled_class_grayscale_means, torch.tensor([background_grayscale_mean], device=image.device)])

        # 保存每个批次的预测和标签灰度均值
        predicted_grayscale_means.append(predicted_class_grayscale_means)
        labeled_grayscale_means.append(labeled_class_grayscale_means)
    stacked_predicted = torch.stack(predicted_grayscale_means, dim=0)
    stacked_labeled = torch.stack(labeled_grayscale_means, dim=0)
    mean_predicted = stacked_predicted.mean(dim=0)
    mean_labeled = stacked_labeled.mean(dim=0)

    # 返回两个灰度值张量，分别是预测和标签的灰度均值
    return mean_predicted,mean_labeled 
