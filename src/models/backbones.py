import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, ConvNeXt_Tiny_Weights

def get_resnet18_backbone():
    """Returns ResNet18 feature extractor (pre-fc)."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    return nn.Sequential(*list(model.children())[:-1])

def get_resnet18_deep_dropout(p=0.15):
    """
    Returns ResNet18 with dropout layers injected at the end of 
    the layer3 and layer4 residual blocks for spatial uncertainty.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # We reconstruct the backbone with internal dropout
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        nn.Dropout2d(p=p), # Spatial dropout at layer 3
        model.layer4,
        nn.Dropout2d(p=p), # Spatial dropout at layer 4
        model.avgpool
    )
    return features

def get_convnext_tiny_backbone():
    """Returns ConvNeXt-Tiny feature extractor (pre-classifier)."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    # ConvNeXt structure: features -> avgpool -> classifier
    # We want features + avgpool
    return nn.Sequential(
        model.features,
        model.avgpool
    )
