"""CNN model architectures for fish biomass estimation.

This module contains various CNN architectures for fish detection, segmentation,
and biomass estimation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connections."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class AttentionBlock(nn.Module):
    """CBAM (Convolutional Block Attention Module) attention mechanism."""
    
    def __init__(self, channels, reduction=16):
        super(AttentionBlock, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        x = x * spatial_att
        
        return x


class FishDetectionCNN(nn.Module):
    """Custom CNN for fish detection and localization.
    
    Outputs bounding box coordinates and confidence scores.
    """
    
    def __init__(self, num_classes=1):
        super(FishDetectionCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Detection heads
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_class = nn.Linear(512, num_classes)  # Classification
        self.fc_bbox = nn.Linear(512, 4)  # Bounding box (x, y, w, h)
        self.fc_conf = nn.Linear(512, 1)  # Confidence score
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and detection
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        class_logits = self.fc_class(x)
        bbox = self.fc_bbox(x)
        confidence = torch.sigmoid(self.fc_conf(x))
        
        return {
            'class_logits': class_logits,
            'bbox': bbox,
            'confidence': confidence
        }


class FishSegmentationCNN(nn.Module):
    """U-Net architecture for fish segmentation.
    
    Standard U-Net with encoder-decoder structure and skip connections.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super(FishSegmentationCNN, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return out


class EnhancedFishSegmentationCNN(nn.Module):
    """U-Net with CBAM attention mechanisms for improved segmentation.
    
    Enhanced U-Net with attention blocks at each decoder level.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super(EnhancedFishSegmentationCNN, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        self.att_bottleneck = AttentionBlock(1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(512)
        self.dec4 = self._conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(256)
        self.dec3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(128)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(64)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with attention
        b = self.bottleneck(self.pool(e4))
        b = self.att_bottleneck(b)
        
        # Decoder with attention and skip connections
        d4 = self.up4(b)
        e4 = self.att4(e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att3(e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att2(e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1 = self.att1(e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.out(d1)
        return out


class FishBiomassCNN(nn.Module):
    """CNN for direct biomass estimation from images.
    
    Regresses biomass value directly from input images.
    """
    
    def __init__(self):
        super(FishBiomassCNN, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Attention
        self.attention = AttentionBlock(512)
        
        # Regression head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)  # Biomass value
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention
        x = self.attention(x)
        
        # Regression
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        biomass = self.fc3(x)
        
        return biomass


class MultiTaskCNN(nn.Module):
    """Multi-task learning CNN for detection, segmentation, and biomass estimation.
    
    Shared encoder with task-specific decoder heads.
    """
    
    def __init__(self, num_classes=1):
        super(MultiTaskCNN, self).__init__()
        
        # Shared encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Detection head
        self.detection_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_bbox = nn.Linear(512, 4)
        self.fc_conf = nn.Linear(512, 1)
        
        # Segmentation head (simplified U-Net decoder)
        self.seg_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_dec1 = self._conv_block(512, 256)
        
        self.seg_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_dec2 = self._conv_block(256, 128)
        
        self.seg_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_dec3 = self._conv_block(128, 64)
        
        self.seg_up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.seg_out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Biomass regression head
        self.biomass_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_biomass1 = nn.Linear(512, 256)
        self.fc_biomass2 = nn.Linear(256, 128)
        self.fc_biomass3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
        # Store encoder features for segmentation
        self.enc_features = {}
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Shared encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        
        # Detection head
        det_x = self.detection_pool(e4)
        det_x = torch.flatten(det_x, 1)
        bbox = self.fc_bbox(det_x)
        confidence = torch.sigmoid(self.fc_conf(det_x))
        
        # Segmentation head
        seg_x = self.seg_up1(e4)
        seg_x = torch.cat([seg_x, e3], dim=1)
        seg_x = self.seg_dec1(seg_x)
        
        seg_x = self.seg_up2(seg_x)
        seg_x = torch.cat([seg_x, e2], dim=1)
        seg_x = self.seg_dec2(seg_x)
        
        seg_x = self.seg_up3(seg_x)
        seg_x = torch.cat([seg_x, e1], dim=1)
        seg_x = self.seg_dec3(seg_x)
        
        seg_x = self.seg_up4(seg_x)
        segmentation = self.seg_out(seg_x)
        
        # Biomass regression head
        bio_x = self.biomass_pool(e4)
        bio_x = torch.flatten(bio_x, 1)
        bio_x = F.relu(self.fc_biomass1(bio_x))
        bio_x = self.dropout(bio_x)
        bio_x = F.relu(self.fc_biomass2(bio_x))
        bio_x = self.dropout(bio_x)
        biomass = self.fc_biomass3(bio_x)
        
        return {
            'bbox': bbox,
            'confidence': confidence,
            'segmentation': segmentation,
            'biomass': biomass
        }


def get_model(model_name, **kwargs):
    """Factory function to get model by name.
    
    Args:
        model_name (str): Name of the model to instantiate
        **kwargs: Additional arguments to pass to model constructor
    
    Returns:
        nn.Module: Instantiated model
    
    Example:
        >>> model = get_model('FishDetectionCNN', num_classes=2)
        >>> model = get_model('FishSegmentationCNN', in_channels=3)
    """
    models = {
        'FishDetectionCNN': FishDetectionCNN,
        'FishSegmentationCNN': FishSegmentationCNN,
        'EnhancedFishSegmentationCNN': EnhancedFishSegmentationCNN,
        'FishBiomassCNN': FishBiomassCNN,
        'MultiTaskCNN': MultiTaskCNN
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == '__main__':
    """Test code for all models."""
    
    print("Testing CNN Models...")
    print("=" * 50)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Test FishDetectionCNN
    print("\n1. Testing FishDetectionCNN...")
    det_model = FishDetectionCNN(num_classes=1)
    det_output = det_model(x)
    print(f"   Class logits shape: {det_output['class_logits'].shape}")
    print(f"   BBox shape: {det_output['bbox'].shape}")
    print(f"   Confidence shape: {det_output['confidence'].shape}")
    
    # Test FishSegmentationCNN
    print("\n2. Testing FishSegmentationCNN...")
    seg_model = FishSegmentationCNN(in_channels=3, num_classes=1)
    seg_output = seg_model(x)
    print(f"   Segmentation output shape: {seg_output.shape}")
    
    # Test EnhancedFishSegmentationCNN
    print("\n3. Testing EnhancedFishSegmentationCNN...")
    enhanced_seg_model = EnhancedFishSegmentationCNN(in_channels=3, num_classes=1)
    enhanced_seg_output = enhanced_seg_model(x)
    print(f"   Enhanced segmentation output shape: {enhanced_seg_output.shape}")
    
    # Test FishBiomassCNN
    print("\n4. Testing FishBiomassCNN...")
    biomass_model = FishBiomassCNN()
    biomass_output = biomass_model(x)
    print(f"   Biomass output shape: {biomass_output.shape}")
    
    # Test MultiTaskCNN
    print("\n5. Testing MultiTaskCNN...")
    multitask_model = MultiTaskCNN(num_classes=1)
    multitask_output = multitask_model(x)
    print(f"   BBox shape: {multitask_output['bbox'].shape}")
    print(f"   Confidence shape: {multitask_output['confidence'].shape}")
    print(f"   Segmentation shape: {multitask_output['segmentation'].shape}")
    print(f"   Biomass shape: {multitask_output['biomass'].shape}")
    
    # Test get_model factory function
    print("\n6. Testing get_model factory function...")
    factory_model = get_model('FishBiomassCNN')
    print(f"   Created model type: {type(factory_model).__name__}")
    
    # Count parameters
    print("\n7. Model Parameter Counts:")
    models_to_test = [
        ('FishDetectionCNN', det_model),
        ('FishSegmentationCNN', seg_model),
        ('EnhancedFishSegmentationCNN', enhanced_seg_model),
        ('FishBiomassCNN', biomass_model),
        ('MultiTaskCNN', multitask_model)
    ]
    
    for model_name, model in models_to_test:
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   {model_name}:")
        print(f"      Total params: {params:,}")
        print(f"      Trainable params: {trainable_params:,}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
