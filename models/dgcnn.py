"""DGCNN classification model for point cloud inputs."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors for each point.

    Args:
        x: Input features of shape (B, C, N).
        k: Number of nearest neighbours.

    Returns:
        Indices of the k nearest neighbours (B, N, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Construct dynamic graph features as proposed in DGCNN."""
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNNCls(nn.Module):
    """Dynamic Graph CNN for classification.

    This implementation follows the architecture from the original DGCNN paper
    (https://arxiv.org/abs/1801.07829) and expects an input tensor of shape
    (B, 3, N).
    """

    def __init__(self, num_classes: int = 40, k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6_layer = self.bn6
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7_layer = self.bn7
        self.linear3 = nn.Linear(256, num_classes)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6_layer(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7_layer(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


def build_dgcnn_classifier(num_classes: int = 40, num_points: int = 1024, k: int = 20, dropout: float = 0.5) -> DGCNNCls:
    """Helper function to build a classification network."""
    _ = num_points  # kept for API compatibility
    return DGCNNCls(num_classes=num_classes, k=k, dropout=dropout)
