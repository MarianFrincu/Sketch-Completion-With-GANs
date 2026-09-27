import torch
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Resnet18Classifier(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self, freeze):
        for param in self.model.parameters():
            param.requires_grad = not freeze

        for param in self.model.fc.parameters():
            param.requires_grad = True


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classifier = Resnet18Classifier(checkpoint['num_classes'], pretrained=False)
    classifier.load_state_dict(checkpoint['state_dict'])
    return classifier.to(device)
