import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VGG(nn.Module) :

    def __init__(self, opts) :
        super(VGG, self).__init__()

        self.opts = opts
        self.build_model()

    def build_model(self) :

        self.use_cuda = self.opts['use_cuda']

        dropout = self.opts['dropout']

        self.encoding_layer = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 64,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 64,
                      out_channels = 128,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 128,
                      out_channels = 256,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=1,
                      padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 256,
                      out_channels = 512,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=1,
                      padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=1,
                      padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.xavier_uniform(m.weight.data)
                m.bias.data.fill_(0.0)

    def compute_loss(self, scores, label) :
        loss = F.cross_entropy(scores, label)
        return loss


    def forward(self, image, label) :
        image = Variable(image)
        label = Variable(label)
        if self.use_cuda :
            image = image.cuda()
            label = label.cuda()
        states = self.encoding_layer(image)
        states = states.contiguous().view(states.size(0), -1)

        scores = self.classifier(states)
        loss = self.compute_loss(scores, label)
        return loss

    def evaluate(self, testloader) :

        total = 0.0
        ans = 0.0
        for image, label in testloader :
            image = Variable(image)
            label = Variable(label)
            label = label.float()
            if self.use_cuda:
                image = image.cuda()
                label = label.cuda()
            states = self.encoding_layer(image)
            states = states.contiguous().view(states.size(0), -1)
            scores = self.classifier(states)
            prediction = torch.max(scores, 1)[1].float()
            compare = torch.eq(prediction, label).float()
            ans += compare.sum(0)
            total += prediction.size(0)

        accuracy = float(ans) / float(total) * 100.0
        print('Accuracy: %.2f %%' % accuracy)
        return accuracy