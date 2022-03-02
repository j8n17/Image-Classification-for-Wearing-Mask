import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchsummary import summary
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

# Identity
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # self.m = timm.create_model('resnet18', pretrained=True)
        # # self.m.layer4 = Identity()
        
        # for name, param in self.m.named_parameters():
        #     if 'layer4' in name:
        #         continue
        #     param.requires_grad = False
        
        # self.m.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )
        self.m = models.resnext50_32x4d(pretrained=True)
        # self.m.fc = nn.Linear(self.m.fc.in_features, num_classes)
        self.m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.m.fc.in_features, num_classes)
        )
        # self.m.classifier[6] = nn.Linear(self.m.classifier[6].in_features, num_classes)
    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.m(x)
        # return self.m(x)

class ResNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = models.resnext50_32x4d(pretrained=True)
        # self.m.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.m.fc.in_features, num_classes)
        # )
    
    def forward(self, x):
        return self.m(x)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = models.googlenet(pretrained = True)#모델에 dropout원래 있음
        self.m.fc = nn.Linear(self.m.fc.in_features, num_classes)

    def forward(self, x):
        return self.m(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = models.resnet18(pretrained = True)
        self.m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.m.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.m(x)

class ShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = models.shufflenet_v2_x1_0(pretrained=True)
        self.m.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(self.m.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.m(x)

class ConvNextTiny_hnf(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = timm.create_model('convnext_tiny_hnf')
        self.m.head.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.m.head.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.m(x)


class vitTiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.m.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.m.head.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.m(x)


class vitBase(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = timm.create_model('vit_base_patch32_224', pretrained=True)
        self.m.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.m.head.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.m(x)