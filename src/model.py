import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.attention_utils import AttentionCropLayer
from torch.autograd import Variable


class RACNN(nn.Module):
    def __init__(self, num_classes):
        super(RACNN, self).__init__()
        
        self.convolution1 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes)
        self.convolution2 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes)
        self.convolution3 = models.efficientnet.efficientnet_v2_s(num_classes=num_classes)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(256, num_classes)
        self.fc3 = nn.Linear(256, num_classes)  # TODO: adjust parameters

        self.mapn = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),  # TODO: adjust parameters
            nn.Tanh(),
            nn.Linear(1024, 6),
            nn.Sigmoid()
        )

        self.crop_resize = AttentionCropLayer()
        self.echo = None
    
    def forward(self, x):
        rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False).cuda()

        feature1 = self.convolution1.features[:-1](x)
        feature_pool1 = self.pool(feature1)
        attentions = self.mapn(feature1.view(-1, 256 * 14 * 14))
        cropped2 = self.crop_resize(x, attentions[:, :3] * rescale_tl * x.shape[-1])  # TODO: rescaling of attentions
        cropped3 = self.crop_resize(x, attentions[:, 3:] * rescale_tl * x.shape[-1])

        feature2 = self.convolution2.features[:-1](cropped2)
        feature3 = self.convolution3.features[:-1](cropped3)
        feature_pool2 = self.pool(feature2)
        feature_pool3 = self.pool(feature3)

        scores1 = self.fc1(feature_pool1.view(-1, 256))  # TODO: modification of input shape
        scores2 = self.fc2(feature_pool2.view(-1, 256))
        scores3 = self.fc3(feature_pool3.view(-1, 256))

        return [scores1, scores2, scores3], feature1, attentions, [cropped2, cropped3]

    def echo_backbone(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.classification_loss(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def echo_mapn(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, attentions, _ = self.forward(inputs)
        optimizer.zero_grad()
        loss = self.rank_loss(logits, targets, attentions)
        loss.backward()
        optimizer.step()
        return loss.item()

    def mode(self, mode_type):
        if mode_type == 'pretrain_mapn':
            self.echo = self.echo_pretrain_mapn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.echo_backbone
            self.train()
        if mode_type == 'mapn':
            self.echo = self.echo_mapn
            self.eval()

    @staticmethod
    def classification_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            loss.append(F.cross_entropy(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss

    @staticmethod
    def rank_loss(logits, targets, attentions, margin=0.05):
        loss = 0

        preds = [F.softmax(x, dim=-1) for x in logits]
        set_pt = [[scaled_pred[batch_inner_id][target] for scaled_pred in preds] for batch_inner_id, target in enumerate(targets)]
        for batch_inner_id, pts in enumerate(set_pt):
            loss += (pts[0] - pts[1] + margin).clamp(min=0)
            loss += (pts[0] - pts[2] + margin).clamp(min=0)

            x1, y1, l1, x2, y2, l2 = attentions[batch_inner_id]
            tl1, br1, tl2, br2 = [x1 - l1, y1 - l1], [x1 + l1, y1 + l1], [x2 - l2, y2 - l2], [x2 + l2, y2 + l2]
            x_dist = (min(br1[0], br2[0]) - max(tl1[0], tl2[0]))
            y_dist = (min(br1[1], br2[1]) - max(tl1[1], tl2[1]))
            if x_dist > 0 and y_dist > 0:
                loss += x_dist * y_dist
        return loss

    @staticmethod
    def get_weak_loc(features):
        resize = 448
        response_map_batch = F.interpolate(features, size=[resize, resize], mode="bilinear").mean(1)
        ret_batch = torch.zeros([len(response_map_batch), 6])
        for i, response_map in enumerate(response_map_batch):
            response_map_copy = torch.clone(response_map)
            argmax_idx = response_map_copy.argmax()
            ty = (argmax_idx % resize)
            argmax_idx = (argmax_idx - ty)/resize
            tx = (argmax_idx % resize)
            cur_batch = [(tx*1.0/resize).clamp(min=0.25, max=0.75), (ty*1.0/resize).clamp(min=0.25, max=0.75), 0.25]  # tl = 0.25, fixed
            tl = [(cur_batch[0] - 0.2) * resize, (cur_batch[1] - 0.2) * resize]
            br = [(cur_batch[0] + 0.2) * resize, (cur_batch[1] + 0.2) * resize]
            response_map_copy[int(tl[0]):int(br[0]), int(tl[1]):int(br[1])] = 0
            argmax_idx2 = response_map_copy.argmax()
            ty2 = (argmax_idx2 % resize)
            argmax_idx2 = (argmax_idx - ty2)/resize
            tx2 = (argmax_idx2 % resize)
            cur_batch.extend([(tx2*1.0/resize).clamp(min=0.25, max=0.75), (ty2*1.0/resize).clamp(min=0.25, max=0.75), 0.25])
            ret_batch[i] = torch.tensor(cur_batch)
        return torch.tensor(ret_batch)

    def echo_pretrain_mapn(self, inputs, optimizer):
        inputs = Variable(inputs).cuda()
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.get_weak_loc(features)
        optimizer.zero_grad()
        print(attens.shape)
        print(weak_loc.shape)
        weak_loss = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        loss = weak_loss
        loss.backward()
        optimizer.step()
        return loss.item()
