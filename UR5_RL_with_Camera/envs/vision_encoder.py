# envs/vision_encoder.py
import torch
import torch.nn as nn

class DepthEncoder(nn.Module):
    """
    小型 CNN，把單通道深度圖壓縮成固定維度特徵向量。
    輸入：[N, 1, H, W]
    輸出：[N, feature_dim]
    """
    def __init__(self, img_size: int = 84, feature_dim: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # [N,32,20,20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # [N,64,9,9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # [N,64,7,7]
            nn.ReLU(),
            nn.Flatten(),                                 # [N, 3136]
        )

        # 自動推算 CNN 輸出維度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))