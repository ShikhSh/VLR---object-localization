import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # TODO (Q1.1): Define model
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

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, x, box_plots=False):
        # TODO (Q1.1): Define forward pass
        feats = self.classifier(self.features(x))
        out = F.adaptive_max_pool2d(feats,output_size=1)
        if box_plots:
            return feats
        return out


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
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

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x, box_plots=False):
        # TODO (Q1.7): Define forward pass
        feats = self.classifier(self.features(x))
        out = F.adaptive_avg_pool2d(feats,output_size=1)
        if box_plots:
            return feats
        return out


def xavier_init(layer):
    # Xavier init of the weights (Q1.1) (Q1.7)
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)

def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    (Q1.1) network init
    Functionality:
    1. Xavier init of the weights of the network
    2. If the pre-trained flag is set to true, then load the pre-trained weights of alexnet for the common layers

    """
    model = LocalizerAlexNet(**kwargs)
    model.apply(xavier_init)
    #TODO: Ignore for now until instructed
    if pretrained:
        # copy all the weights that can be copied from the pre-trained AlexNet
        weights = model_zoo.load_url(model_urls['alexnet'], 'model/')
        own_state = model.state_dict()

        for name, param in weights.items():
            print(name)
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            try:
                print(param.shape)
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue
        model.load_state_dict(own_state)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    (Q1.7) network init 
    Functionality:
    1. Xavier init of the weights of the network
    2. If the pre-trained flag is set to true, then load the pre-trained weights of alexnet for the common layers

    """
    model = LocalizerAlexNetRobust(**kwargs)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    model.apply(xavier_init)
    if pretrained:
        # copy all the weights that can be copied from the pre-trained AlexNet
        weights = model_zoo.load_url(model_urls['alexnet'], 'model/')

        own_state = model.state_dict()

        for name, param in weights.items():
            print(name)
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue
        model.load_state_dict(own_state)

    return model
