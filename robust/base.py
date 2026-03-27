from copy import deepcopy
class CleanGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, data, label):
        return data

class BaseAdvDataGenerator:
    def __init__(self, attacker):
        self.attacker=attacker

    def __iter__(self):
        return iter(self.attacker)

    def generate(self, data, label):
        return self.attacker.attack(data, label)
    
    def forward(self,images, labels, loss):
        return self.attacker.step(images, labels, loss)

class AdvDataGenerator(BaseAdvDataGenerator):
    def __init__(self, attacker, loss):
        super().__init__(attacker)
        self.attacker.set_loss(loss=loss)

class CLS_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker,criterion)
        self.loss_cls=lambda  preds, batch: self.criterion(preds, batch)[1]
        self.attacker = attacker
        self.criterion = criterion
        
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
        img_cls = self.attacker.step(images, boxes, self.loss_cls)
        img_cls = img_cls.to(device)
        return img_cls 

class LOC_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion):
        super().__init__(attacker,criterion)
        self.loss_loc = lambda  preds, batch: self.criterion(preds, batch)[0]
        self.attacker = attacker
        self.criterion = criterion
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
        img_loc = self.attacker.step(images, boxes, self.loss_loc)
        img_loc = img_loc.to(device)
        return img_loc

class CON_ADG(AdvDataGenerator):
    def __init__(self, attacker, criterion, rate=[1,1,1]):
        super().__init__(attacker,criterion)
        self.criterion = criterion
        self.rate_1 = 1
        self.rate_2 = 1
        self.rate_3 = 1
        self.loss=lambda  preds, batch: (self.criterion(preds, batch)[0]*self.rate_1+self.criterion(preds, batch)[1]*self.rate_2+self.criterion(preds, batch)[2]*self.rate_3)
        # self.loss_loc=lambda  imgs, boxes,labels: self.criterion(imgs, boxes,labels)[0]
        # self.loss_cls=lambda  imgs, boxes,labels: self.criterion(imgs, boxes,labels)[1]
        self.attacker = attacker
        
    def __call__(self, images, boxes, model_train):
        return self.forward(images, boxes, model_train)
    def forward(self, images, boxes,model_train):
        device = images.device
        img =  self.attacker.step(images, boxes, self.loss)
        img = img.to(device)
        return img