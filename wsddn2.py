import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN2(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=classes):
        super(WSDDN2, self).__init__()

        print(f'classes {classes}')

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True)
        )
        self.roi_pool = roi_pool
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.score_fc   = nn.Linear(4096, len(classes))
        self.bbox_fc    = nn.Linear(4096, len(classes))

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        # image: [1, 3, 512, 512]

        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        inp_size = image.shape[-1]*1.0
        x = self.features(image)
        feature_dim = x.shape[-1]*1.0

        # print(f'rois.shape {rois.shape}')
        rois = rois.squeeze()
        # [rois.type('torch.FloatTensor').cuda()]
        x = self.roi_pool(x, [rois.type('torch.FloatTensor').cuda()], output_size=6, spatial_scale=feature_dim/inp_size) # 300x256x7x7 [N_roi X C X OUT X OUT]
        x = x.flatten(start_dim=1)
        x = self.classifier(x) # (N_roi X 4096)
        
        class_score = self.score_fc(x) # (N_roi X 20)
        bbox = self.bbox_fc(x) # (N_roi X 20)
        
        class_score = torch.softmax(class_score, dim = 1) # (N_roi X 20)
        bbox = torch.softmax(bbox, dim = 0) # (N_roi X 20)

        cls_prob = class_score*bbox # (N_roi X 20) [for each roi we have a prob of that roi belonging to a particular class]


        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        cls_prob = torch.sum(cls_prob, dim=0).unsqueeze(1)
        cls_prob = torch.clamp(cls_prob, 0., 1.)
        print("HEY THERE::::")
        print(cls_prob)
        print(label_vec)
        loss = F.binary_cross_entropy(cls_prob, label_vec, reduction='sum')
        return loss