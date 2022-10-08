import torch.nn as nn
import torchvision.models as models
import torch


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
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

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )

        self.classifier.apply(self.initialize_sequential)
    
    def initialize_sequential(self, l):
        if isinstance(l, nn.Conv2d):
            # nn.init.xavier_uniform(l.weight)
            nn.init.xavier_normal_(l.weight.data)
            # torch.nn.init.kaiming_uniform(m.weight)
            l.bias.data.fill_(0.00)


    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        out = self.features(x)
        out = self.classifier(out)
        out = torch.max(out, dim = 2)[0].max(2)[0]
        return out


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
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

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
        )

        self.classifier.apply(self.initialize_sequential)
    
    def initialize_sequential(self, l):
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight.data)
            # torch.nn.init.kaiming_uniform(m.weight)
            l.bias.data.fill_(0.00)

    def forward(self, x):
        # TODO (Q1.7): Define forward pass

        out = self.features(x)
        out = self.classifier(out)
        out = torch.mean(out, dim = 2)[0].mean(2)[0]
        return out


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    if pretrained:
        alex_net_pretrained = models.alexnet(pretrained = True)
        for i in [0, 3, 6, 8, 10]:
            model.features[i].load_state_dict(alex_net_pretrained.features[i].state_dict())

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    
    if pretrained:
        alex_net_pretrained = models.alexnet(pretrained = True)
        for i in [0, 3, 6, 8, 10]:
            model.features[i].load_state_dict(alex_net_pretrained.features[i].state_dict())
    
    return model
