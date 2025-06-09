import torch
import torch.nn as nn
import torch_geometric
import torch_scatter
    
from trimesh import transformations
import math
import numpy as np
from torch import nn

import torch.nn.functional as F
from time import time
from util.positional_encoding import get_embedder


class SimpleMLPEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedder, self.embed_dim = get_embedder(10)
        self.fc1 = nn.Linear(self.embed_dim * 3 + 7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

    def forward(self, x, _batch):
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9])
        x = torch.cat([x_0, x_1, x_2, x[:, 9:]], dim=-1)
        # 16 -> 64
        x = self.relu(self.norm1(self.fc1(x)))
        # 64 -> 128
        x = self.relu(self.norm2(self.fc2(x)))
        point_features = x
        # 128 -> 512
        x = self.norm3(self.fc3(x))
        return torch.cat([x, point_features], dim=-1)

class GraphEncoder(nn.Module):

    def __init__(self, no_max_pool=True, aggr='mean', graph_conv="edge", use_point_features=False, output_dim=512, order_invariant=False, semantic_features=False, geometry_features=False, articulation_features=False, pc_head=False):
        super().__init__()
        self.no_max_pool = no_max_pool
        self.use_point_features = use_point_features
        self.order_invariant = order_invariant
        self.embedder, self.embed_dim = get_embedder(10)
        self.conv = graph_conv
        
        self.proj_semantic = semantic_features
        self.proj_geometry = geometry_features
        self.proj_articulation = articulation_features
        self.proj_pc = pc_head
        if self.proj_semantic and self.proj_geometry and self.proj_articulation:
            self.gc1 = get_conv(self.conv, self.embed_dim * 4 + 7 + 32 + 128 + 32 + 32, 64, aggr=aggr)
        elif self.proj_semantic and self.proj_articulation and self.proj_pc:
            self.gc1 = get_conv(self.conv, self.embed_dim * 4 + 7 + 32 + 32*4, 64, aggr=aggr)
        elif self.proj_semantic and self.proj_geometry and self.proj_pc:
            self.gc1 = get_conv(self.conv, self.embed_dim * 6 + 7 + 32 + 128, 64, aggr=aggr)
        elif self.proj_semantic and self.proj_geometry:
            self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7 + 32 + 128, 64, aggr=aggr)
        elif self.proj_semantic and self.proj_pc:
            self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7 + 32 + 3*32, 64, aggr=aggr)
        elif self.proj_semantic and self.proj_articulation:
            self.gc1 = get_conv(self.conv, self.embed_dim * 4 + 7 + 32 + 32 + 32, 64, aggr=aggr)
        elif self.proj_semantic:
            self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7 + 32, 64, aggr=aggr)
        else:
            self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7, 64, aggr=aggr)
        self.gc2 = get_conv(self.conv, 64, 128, aggr=aggr)
        self.gc3 = get_conv(self.conv, 128, 256, aggr=aggr)
        self.gc4 = get_conv(self.conv, 256, 256, aggr=aggr)
        self.gc5 = get_conv(self.conv, 256, output_dim, aggr=aggr)

        self.norm1 = torch_geometric.nn.BatchNorm(64)
        self.norm2 = torch_geometric.nn.BatchNorm(128)
        self.norm3 = torch_geometric.nn.BatchNorm(256)
        self.norm4 = torch_geometric.nn.BatchNorm(256)

        self.relu = nn.ReLU()
        if self.proj_semantic:
            self.semantic_features_projection = nn.Linear(768, 32)
            
        if self.proj_geometry:
            self.geometry_features_projection = nn.Linear(128, 128)
            
        if self.proj_pc:
            self.pc_feat_projection = nn.Linear(128, 32)
        
        if self.proj_articulation: # joint location, orientation, type (text feature), joint limit
            self.embedder_joint_location, self.embed_dim_joint_location = get_embedder(10)
            self.articulation_joint_orientation_projection = nn.Linear(3, 32)
            self.articulation_joint_semantic_features_projection = nn.Linear(768, 32)

    def forward(self, x, edge_index, batch, semantic_features=None, geometry_features=None, articulation_features=None, pc=None, pc_feat=None):
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9])
        x_n = x[:, 9:12]
        x_ar = x[:, 12:13]
        x_an_0 = x[:, 13:14]
        x_an_1 = x[:, 14:15]
        x_an_2 = x[:, 15:]
        if not self.order_invariant:
            if self.proj_semantic:
                s = self.semantic_features_projection(semantic_features)
            if self.proj_geometry:
                g = self.geometry_features_projection(geometry_features)
            if self.proj_articulation:
                j_l = self.embedder_joint_location(articulation_features[:, :3])
                j_o = self.articulation_joint_orientation_projection(articulation_features[:, 3:6])
                j_t = self.articulation_joint_semantic_features_projection(articulation_features[:, 9:])
            if self.proj_pc:
                x_pc_0 = self.pc_feat_projection(pc[:,:128])
                x_pc_1 = self.pc_feat_projection(pc[:,128:256])
                x_pc_2 = self.pc_feat_projection(pc[:,256:])
            if self.proj_semantic and self.proj_geometry and self.proj_articulation:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, g, j_l, j_o, j_t], dim=-1)
            elif self.proj_semantic and self.proj_geometry and self.proj_pc:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, g, x_pc_0, x_pc_1, x_pc_2], dim=-1)
            elif self.proj_semantic and self.proj_articulation and self.proj_pc:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, x_pc_0, x_pc_1, x_pc_2, j_l, j_o], dim=-1)
            elif self.proj_semantic and self.proj_geometry:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, g], dim=-1)
            elif self.proj_semantic and self.proj_pc:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, x_pc_0, x_pc_1, x_pc_2], dim=-1)
            elif self.proj_semantic and self.proj_articulation:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s, j_l, j_o, j_t], dim=-1)
            elif self.proj_semantic:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2, s], dim=-1)
            else:
                x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2], dim=-1)
            x = self.relu(self.norm1(self.gc1(x, edge_index)))
        else:
            x_n = torch.zeros_like(x_n)  # zero out normals for order invariance
            x_c0 = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2], dim=-1)
            x_c1 = torch.cat([x_0, x_2, x_1, x_n, x_ar, x_an_0, x_an_2, x_an_1], dim=-1)
            x_c2 = torch.cat([x_1, x_0, x_2, x_n, x_ar, x_an_1, x_an_0, x_an_2], dim=-1)
            x_c3 = torch.cat([x_1, x_2, x_0, x_n, x_ar, x_an_1, x_an_2, x_an_0], dim=-1)
            x_c4 = torch.cat([x_2, x_0, x_1, x_n, x_ar, x_an_2, x_an_0, x_an_1], dim=-1)
            x_c5 = torch.cat([x_2, x_1, x_0, x_n, x_ar, x_an_2, x_an_1, x_an_0], dim=-1)
            x_c0 = self.relu(self.norm1(self.gc1(x_c0, edge_index)))
            x_c1 = self.relu(self.norm1(self.gc1(x_c1, edge_index)))
            x_c2 = self.relu(self.norm1(self.gc1(x_c2, edge_index)))
            x_c3 = self.relu(self.norm1(self.gc1(x_c3, edge_index)))
            x_c4 = self.relu(self.norm1(self.gc1(x_c4, edge_index)))
            x_c5 = self.relu(self.norm1(self.gc1(x_c5, edge_index)))
            x = (x_c0 + x_c1 + x_c2 + x_c3 + x_c4 + x_c5) / 6
        x = self.norm2(self.gc2(x, edge_index))
        point_features = x
        x = self.relu(x)
        x = self.relu(self.norm3(self.gc3(x, edge_index)))
        x = self.relu(self.norm4(self.gc4(x, edge_index)))
        x = self.gc5(x, edge_index)
            
        if not self.no_max_pool:
            x = torch_scatter.scatter_max(x, batch, dim=0)[0]
            x = x[batch, :]
        if self.use_point_features:
            return torch.cat([x, point_features], dim=-1)
        return x


class GraphEncoderTriangleSoup(nn.Module):

    def __init__(self, aggr='mean', graph_conv="edge"):
        super().__init__()
        self.embedder, self.embed_dim = get_embedder(10)
        self.conv = graph_conv
        self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7, 96, aggr=aggr)
        self.gc2 = get_conv(self.conv, 96, 192, aggr=aggr)
        self.gc3 = get_conv(self.conv, 192, 384, aggr=aggr)
        self.gc4 = get_conv(self.conv, 384, 384, aggr=aggr)
        self.gc5 = get_conv(self.conv, 384, 576, aggr=aggr)

        self.norm1 = torch_geometric.nn.BatchNorm(96)
        self.norm2 = torch_geometric.nn.BatchNorm(192)
        self.norm3 = torch_geometric.nn.BatchNorm(384)
        self.norm4 = torch_geometric.nn.BatchNorm(384)

        self.relu = nn.ReLU()

    @staticmethod
    def distribute_features(features, face_indices, num_vertices):
        N, F = features.shape
        features = features.reshape(N * 3, F // 3)
        assert features.shape[0] == face_indices.shape[0] * face_indices.shape[1], "Features and face indices must match in size"
        vertex_features = torch.zeros([num_vertices, features.shape[1]], device=features.device)
        torch_scatter.scatter_mean(features, face_indices.reshape(-1), out=vertex_features, dim=0)
        distributed_features = vertex_features[face_indices.reshape(-1), :]
        distributed_features = distributed_features.reshape(N, F)
        return distributed_features

    def forward(self, x, edge_index, faces, num_vertices):
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9])
        x = torch.cat([x_0, x_1, x_2, x[:, 9:]], dim=-1)
        x = self.relu(self.norm1(self.gc1(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm2(self.gc2(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm3(self.gc3(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm4(self.gc4(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.gc5(x, edge_index)
        x = self.distribute_features(x, faces, num_vertices)
        return x


def get_conv(conv, in_dim, out_dim, aggr):
    if conv == 'sage':
        return torch_geometric.nn.SAGEConv(in_dim, out_dim, aggr=aggr)
    elif conv == 'gat':
        return torch_geometric.nn.GATv2Conv(in_dim, out_dim, fill_value=aggr)
    elif conv == 'edge':
        return torch_geometric.nn.EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_dim * 2, 2 * out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * out_dim, out_dim),
            ),
            aggr=aggr,
        )


class ConvNetDecoder(nn.Module):

    def __init__(self, num_tokens=257, ce_output=True, use_point_features=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.ce_output = ce_output
        if use_point_features:
            self.fc1 = nn.Conv1d(512 + 128, 512, 5, padding=2)
        else:
            self.fc1 = nn.Conv1d(512, 512, 5, padding=2)
        self.fc2 = nn.Conv1d(512, 384, 5, padding=2)
        self.fc3 = nn.Conv1d(384, 384, 5, padding=2)
        self.fc4 = nn.Conv1d(384, 384, 3, padding=1)
        if ce_output:
            self.fc5 = nn.Conv1d(384, self.num_tokens * 9, 3, padding=1)
        else:
            self.fc5 = nn.Conv1d(384, 9, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(384)
        self.norm3 = nn.BatchNorm1d(384)
        self.norm4 = nn.BatchNorm1d(384)

        self.relu = nn.ReLU()

    def forward(self, x):
        B, _, N = x.shape
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.relu(self.norm3(self.fc3(x)))
        x = self.relu(self.norm4(self.fc4(x)))
        x = self.fc5(x)
        if self.ce_output:
            x = x.permute((0, 2, 1)).reshape(B, N, 9, self.num_tokens)
            return x
        return x
    



def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

    
class PointNet(nn.Module):

    def __init__(self, num_class, normal_channel=True):
        super(PointNet, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, return_feats=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        feats = x
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        if return_feats:
            return x, feats
        return x, l3_points
    
    @torch.no_grad()
    def classifier_guided_filter(self, mesh, expected_category):
        interesting_categories = {
            '03001627': ([8, 30], 0.05),
            '04379243': ([33, 3], 0.03),
        }
        category = interesting_categories[expected_category][0]
        threshold = interesting_categories[expected_category][1]
        points = get_point_cloud(mesh, nvotes=8).cuda()
        points = points.transpose(2, 1)
        pred, _ = self(points)
        pred = pred.mean(dim=0)
        pred_probability = torch.nn.functional.softmax(pred.unsqueeze(0), dim=-1)[0]
        pval = pred_probability[category].max().item()
        return pval > threshold
    
    
def get_pointnet_classifier(ckpt_path):
    classifier = PointNet(40, normal_channel=False)
    checkpoint = torch.load(ckpt_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    return classifier


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_point_cloud(mesh, nvotes=4):
    point_batch = []
    for _ in range(nvotes):
        tmesh = mesh
        rot_matrix = transformations.rotation_matrix(-math.pi / 2, [1, 0, 0], [0, 0, 0])
        tmesh = tmesh.apply_transform(rot_matrix)
        rot_matrix = transformations.rotation_matrix(-math.pi / 2, [0, 1, 0], [0, 0, 0])
        tmesh = tmesh.apply_transform(rot_matrix)
        point_set = pc_normalize(tmesh.sample(1024))
        point_batch.append(point_set[np.newaxis, :, :])
    point_batch = torch.from_numpy(np.concatenate(point_batch, axis=0)).float()
    return point_batch