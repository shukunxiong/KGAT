from matplotlib.pyplot import box
from .base import BaseAdvDataGenerator
from copy import deepcopy
import torch

class MTD(BaseAdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker)
        self.criterion = criterion
        self.loss_box=lambda  preds, batch: self.criterion(preds, batch)[0]
        self.loss_cls=lambda  preds, batch: self.criterion(preds, batch)[1]
        self.loss_dfl=lambda  preds, batch: self.criterion(preds, batch)[2]
        self.attacker = attacker
    def __call__(self, images, boxes, model_train):
        return self.forward(images, boxes, model_train)
        # def mtd_forward(attacker, images, labels):
        #     img_loc=attacker.step(deepcopy(images), labels, self.loss_loc)
        #     img_cls=attacker.step(images, labels, self.loss_cls)
        #     with torch.no_grad():
        #         img=img_loc if sum(self.criterion(attacker.model(img_loc), labels)) > sum(
        #             self.criterion(attacker.model(img_cls), labels)) else img_cls
        #     return img

        # self.attacker.set_forward(mtd_forward)
    
    def forward(self, images, boxes,model_train):
        device = images.device
        img_box = self.attacker.step(images, boxes, self.loss_box)
        img_cls = self.attacker.step(images, boxes, self.loss_cls)
        img_box = img_box.to(device)
        img_cls = img_cls.to(device)
        with torch.no_grad():
            img=img_box if sum(self.criterion(model_train(img_box), boxes)) > sum(
                self.criterion(model_train(img_cls), boxes)) else img_cls
        return img
        
        
        # return super().forward(images, labels, loss)