import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Attention(nn.Module):
    def __init__(self, n_class=2, bn_track_running_stats=True):
        super(Attention, self).__init__()
        self.L = 64 
        self.D = 32 
        self.K = 1

        resnet = models.resnet18( pretrained=True )
        # Since patches in each batch belong to a WSI, switching off batch statistics tracking
        # Or reinitializing batch parameters and changing momentum for quick domain adoption
        #if bn_track_running_stats:
        #    for modules in resnet.modules():
        #        if isinstance( modules, nn.BatchNorm2d ):
        #            modules.track_running_stats = False
        #else:
        #    for modules in resnet.modules():
        #        if isinstance( modules, nn.BatchNorm2d ):
        #            modules.momentum = 0.9
        #            modules.weight = nn.Parameter( torch.ones( modules.weight.shape ) )
        #            modules.running_mean = torch.zeros( modules.weight.shape )
        #            modules.bias = nn.Parameter( torch.zeros( modules.weight.shape ) )
        #            modules.running_var = torch.ones( modules.weight.shape )

        modules = list(resnet.children())[:-1]
        self.resnet_head = nn.Sequential(*modules)


        self.resnet_tail = nn.Sequential(nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.L),
                                        nn.ReLU())

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Tanh(),
                                        nn.Linear(self.D, self.K))

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_class)
        )

    def forward(self, x):
        # print('========================')
        # print(x.shape)
        x = x.squeeze(0)
        # print(x.shape)
        # print('========================')
        H = self.resnet_head(x)
        #bs,50,4,4
        #H = H.view(-1, 50 * 4 * 4)
        H = H.view(H.size(0), -1)
        H = self.resnet_tail(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return M, Y_prob


class EncAttn(nn.Module):
    def __init__(self, model_base):
        super(EncAttn, self).__init__()
        self.resnet_head = model_base.resnet_head
        self.resnet_tail = model_base.resnet_tail
        self.attention = model_base.attention

    def forward(self, x):
        x = x.squeeze(0)
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)
        attn = self.attention(x)

        return attn, x
