import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.attention_utils import AttentionCropLayer
from torch.autograd import Variable


class RACNN(nn.Module):
    def __init__(self, num_classes):
        super(RACNN, self).__init__()
        
        self.convolution1 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes).features
        self.convolution2 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes).features
        self.convolution3 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes).features

        self.classification1 = nn.Sequential(nn.Linear(256, num_classes), nn.Softmax(dim=1))
        self.classification2 = nn.Sequential(nn.Linear(256, num_classes), nn.Softmax(dim=1))
        self.classification3 = nn.Sequential(nn.Linear(256, num_classes), nn.Softmax(dim=1))

        self.mapn = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 6),
            nn.Sigmoid()
        )

        self.crop_resize = AttentionCropLayer()
        self.echo = None
    
    def forward(self, x):
        rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False).cuda()

        feature1 = self.convolution1(x)
        attentions = self.mapn(feature1)
        cropped2 = self.crop_resize(x, attentions[:3] * rescale_tl * x.shape[-1])  # TODO: rescaling of attentions
        cropped3 = self.crop_resize(x, attentions[3:] * rescale_tl * x.shape[-1])

        feature2 = self.convolution2(cropped2)
        feature3 = self.convolution3(cropped3)

        # pool_s1 = self.feature_pool1(feature_s1)
        # attention_s1 = self.apn1(feature_s1.view(-1, 256 * 14 * 14))*rescale_tl
        # resized_s1 = self.crop_resize(x, attention_s1 * x.shape[-1])

        # forward @scale-2
        # feature_s2 = self.convolution2.features[:-1](resized_s1)
        # pool_s2 = self.feature_pool2(feature_s2)
        # attention_s2 = self.apn2(feature_s2.view(-1, 256 * 7 * 7))*rescale_tl
        # resized_s2 = self.crop_resize(resized_s1, attention_s2 * resized_s1.shape[-1])

        # forward @scale-3
        # feature_s3 = self.convolution3.features[:-1](resized_s2)
        # pool_s3 = self.feature_pool2(feature_s3)

        scores1 = self.classification1(feature1.view(-1, 256))  # TODO: modification of input shape
        scores2 = self.classification2(feature2.view(-1, 256))
        scores3 = self.classification3(feature3.view(-1, 256))

        return [scores1, scores2, scores3], [cropped2, cropped3]

    def echo_pretrain_apn(self, inputs, optimizer):
        inputs = Variable(inputs).cuda()
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.get_weak_loc(features)
        optimizer.zero_grad()
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1].cuda())
        loss = weak_loss1 + weak_loss2
        loss.backward()
        optimizer.step()
        return loss.item()

    def echo_backbone(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.multitask_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def echo_apn(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.rank_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def mode(self, mode_type):
        assert mode_type in ['pretrain_apn', 'apn', 'backbone']
        if mode_type == 'pretrain_apn':
            self.echo = self.echo_pretrain_apn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.echo_backbone
            self.train()
        if mode_type == 'apn':
            self.echo = self.echo_apn
            self.eval()

    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            loss.append(F.cross_entropy(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss

    @staticmethod
    def rank_loss(logits, targets, margin=0.05):
        preds = [F.softmax(x, dim=-1) for x in logits]
        set_pt = [[scaled_pred[batch_inner_id][target] for scaled_pred in preds] for batch_inner_id, target in enumerate(targets)]
        loss = 0
        for batch_inner_id, pts in enumerate(set_pt):
            loss += (pts[0] - pts[1] + margin).clamp(min=0)
            loss += (pts[1] - pts[2] + margin).clamp(min=0)
        return loss

    @staticmethod
    def get_weak_loc(features):
        ret = []
        for i in range(len(features)):
            resize = 224 if i >= 1 else 448
            response_map_batch = F.interpolate(features[i], size=[resize, resize], mode="bilinear").mean(1)
            ret_batch = []
            for response_map in response_map_batch:
                argmax_idx = response_map.argmax()
                ty = (argmax_idx % resize)
                argmax_idx = (argmax_idx - ty)/resize
                tx = (argmax_idx % resize)
                ret_batch.append([(tx*1.0/resize).clamp(min=0.25, max=0.75), (ty*1.0/resize).clamp(min=0.25, max=0.75), 0.25])  # tl = 0.25, fixed
            ret.append(torch.Tensor(ret_batch))
        return ret
