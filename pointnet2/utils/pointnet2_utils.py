""" PointNet++ utils
Modified by Zenn
Date: Feb 2021
"""
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
from pointnet2.utils import pytorch_utils as pt_utils

from pointnet2.utils.ops import fps as furthest_point_sample
from pointnet2.utils.ops import gather_points as gather_operation
from pointnet2.utils.ops import three_nn as three_nn
from pointnet2.utils.ops import three_interpolate as three_interpolate
from pointnet2.utils.ops import gather_points as grouping_operation
from pointnet2.utils.ops import ball_query as ball_query

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, return_idx=False, normalize_xyz=False):
        # type: (QueryAndGroup, float, int, bool,bool,bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.return_idx = return_idx
        self.normalize_xyz = normalize_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        # instead ball_query
        idx = ball_query(xyz, new_xyz, self.radius, self.nsample)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2).contiguous()
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            features_t = features.transpose(1, 2).contiguous()
            grouped_features = grouping_operation(features_t, idx)
            grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        if self.return_idx:
            return new_features, idx
        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


def knn_point(k, points1, points2):
    """
    find for each point in points1 the knn in points2
    Args:
        k: k for kNN
        points1: B x npoint1 x d
        points2: B x npoint2 x d

    Returns:
        top_k_neareast_idx: (batch_size, npoint1, k) int32 array, indices to input points
    """
    dist_matrix = torch.cdist(points1, points2)  # B, npoint1, npoint2
    top_k_neareast_idx = torch.argsort(dist_matrix, dim=-1)[:, :, :k]  # B, npoint1, K
    top_k_neareast_idx = top_k_neareast_idx.int().contiguous()
    return top_k_neareast_idx
