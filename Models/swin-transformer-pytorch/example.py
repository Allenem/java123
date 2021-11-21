import torch
from swin_transformer_pytorch import SwinTransformer

net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)
print(net)
dummy_x = torch.randn(2, 3, 224, 224)
print(dummy_x)
print(dummy_x.shape)
logits = net(dummy_x)  # (2, 3)
print(logits)
print(logits.shape)