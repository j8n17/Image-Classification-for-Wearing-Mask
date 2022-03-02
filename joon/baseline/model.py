import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


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


class Preresnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

class densenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.densenet = models.densenet161(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes) #bias=False?

    def forward(self, x):
        return self.densenet(x)


class timm_resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = timm.create_models("resnet18", pretrained=True, num_classes=num_classes)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class SwinNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=num_classes)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class ViT_base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)
    
    def forward(self, x):
        return self.model(x)

class ViT_Large(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_large_patch16_384', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)
    
    def forward(self, x):
        return self.model(x)

class ViT_with_ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_large_patch16_384', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(32, 32)
        return x

class SENet154(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('senet154')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)
    
    def forward(self, x):
        return self.model(x)



