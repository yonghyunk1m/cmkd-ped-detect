from collections import OrderedDict
import torch
import torch.nn as nn

VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/" \
                 "releases/download/v0.1/vggish-10086976.pth"

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layer_1s():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M5", 512, 512, "M5"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "M5":
            layers += [nn.MaxPool2d(kernel_size=(2, 5), stride=(2, 5))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGish(nn.Module):
    def __init__(self, pretrained=True, freeze=False, per_second=False) -> None:
        super().__init__()
        self.output_channel = 512
        self.model = VGG(make_layers()) if not per_second else VGG(make_layer_1s())
        
        if pretrained:
            self.load_pretrained_model()
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Input: [Batch, 64, Seq_Len] (Mel-Spectrogram)
        Output: [Batch, 512, T_new] (Acoustic Embeddings)
        """
        if x.dim() == 3:
            x = x.unsqueeze(dim=1)  # (N, 1, F, T)
        
        x = self.model(x) # [N, 512, F_reduced, T_reduced]
    
        x = torch.mean(x, dim=2)  # [N, 512, T_reduced]
        return x

    def load_pretrained_model(self):
        state_dict = torch.hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("features"):
                new_state_dict[key] = value
        self.model.load_state_dict(new_state_dict)