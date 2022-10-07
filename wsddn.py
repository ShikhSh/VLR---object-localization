import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda")

ROI_OUTPUT_DIM = 6

class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None, top_n = 300):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
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
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),#256*ROI_OUTPUT_DIM*ROI_OUTPUT_DIM, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.top_n = top_n

        # basically, the output dimensions of this is number of classes because we feed in ROIS for an image,
        # thus, the batch here is anologous to rois
        # thus the output from this rois layer would finally be N_rois * classes, because the input sequence length is N_rois long
        self.score_fc = nn.Sequential(
            nn.Linear(4096, self.n_classes) #self.top_n*self.n_classes),
        )
        self.bbox_fc = nn.Sequential(
            nn.Linear(4096, self.n_classes) #self.top_n*self.n_classes),
        )

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self, image, rois=None, target_labels=None):
        """Define the forward function for 1 image

        :image: image_size x image_size image
        :rois: N_rois x 4 boxes with their positions
        :gt_vec: N_gt(if padded - 42) x 4, again boxes with positions
        
        :returns: N_rois x n_class vector, where the class score for each vector is defined

        """
        
        # TODO (Q2.1): Use image and rois as input
        features = self.features(image)
        # print(features.shape)
        input_dims = image.shape[-1]
        feat_dims = features.shape[-1]
        
        # print("printing_rois_shape",str(rois.shape))
        # print("roisssss")
        # print(features.shape)
        # print(len(rois))
        # print(rois[0].shape)
        rois = rois.squeeze()
        roi_features = self.roi_pool(features, boxes = [rois.type('torch.FloatTensor').cuda()], output_size = (ROI_OUTPUT_DIM,ROI_OUTPUT_DIM), spatial_scale = 1.0*feat_dims/input_dims)
        # print(roi_features.shape)
        flattened_features = torch.flatten(roi_features, start_dim=1)
        # print(flattened_features.shape)
        lin_model_out = self.classifier(flattened_features.cuda())
        score1 = F.softmax(self.bbox_fc(lin_model_out), dim = 0)
        score2 = F.softmax(self.score_fc(lin_model_out), dim = 1)

        cls_prob = torch.mul(score1, score2).clamp(0, 1)


        if self.training:
            label_vec = target_labels.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)

        # return cls_prob which are N_roi X 20 scores
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss
        The loss is computed using the sum of class probs over ROIs and the labels

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        
        # we find the sum of the class prob for each region of interest for an image
        # essentially we would have the sum of probabilities for each of classes
        # calculate the BCE loss on the label vector and the probabilities
        # BCE loss can handle he probabilities, 
        cls_prob = torch.reshape(torch.sum(cls_prob, dim=0),(-1,1))
        # print(cls_prob)
        # print(label_vec)
        # print("printiingLOSSSSS:::::::::")
        # print(cls_prob.shape)
        # print(label_vec.shape)
        # cls_prob = F.softmax(cls_prob)
        loss = F.binary_cross_entropy(cls_prob, label_vec, reduction='sum')#.to(device)

        return loss

# Caveats:
# N/A: Used Softmax in build loss, can try clamp too -> removed since i am clamping the outputs in the forward function itself
