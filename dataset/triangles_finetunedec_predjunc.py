from typing import Mapping, Sequence

import omegaconf
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset, default_collate
from pathlib import Path
import torch.utils.data
import pickle

from torch_geometric.data.data import BaseData
from tqdm import tqdm

from dataset.quantize_and_tokenize_soup import sort_vertices_and_faces, quantize_coordinates, sort_vertices_and_faces_wfacelabels, sort_vertices_and_faces_womerge
from util.misc import normalize_vertices, scale_vertices, shift_vertices, scale_vertices_returnscale, shift_vertices_returnshifts, normalize_vertices_returnbounds
from torch_geometric.data import Dataset as GeometricDataset, Batch
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.loader.dataloader import Collater as GeometricCollator
import networkx as nx

class TriangleNodes(GeometricDataset):

    def __init__(self, config, split, scale_augment, shift_augment, force_category, use_start_stop=False, only_backward_edges=False, face_order_augment=False):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        self.low_augment = config.low_augment
        self.use_start_stop = use_start_stop
        self.ce_output = config.ce_output
        self.face_order_augment = face_order_augment
        self.only_backward_edges = only_backward_edges
        self.num_tokens = config.num_tokens - 3
        self.all_parts_per_epoch = config.all_parts_per_epoch
        
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if force_category is not None:
                for s in ['train', 'val']:
                    data[f'vertices_{s}'] = [data[f'vertices_{s}'][i] for i in range(len(data[f'vertices_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'faces_{s}'] = [data[f'faces_{s}'][i] for i in range(len(data[f'faces_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'face_labels_{s}'] = [data[f'face_labels_{s}'][i] for i in range(len(data[f'face_labels_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'name_{s}'] = [data[f'name_{s}'][i] for i in range(len(data[f'name_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                
                if len(data[f'vertices_val']) == 0:
                    data[f'vertices_val'] = data[f'vertices_train']
                    data[f'faces_val'] = data[f'faces_train']
                    data[f'face_labels_val'] = data[f'face_labels_train']
                    data[f'name_val'] = data[f'name_train']
                    
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
                self.cached_face_labels = data[f'face_labels_{split}']
                
            else:
                multiplier = 16 if split == 'val' else 512
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier
                self.cached_face_labels = data[f'face_labels_train'][:1] * multiplier

        if self.all_parts_per_epoch:
            # each epoch should contain all parts of all shapes
            self.meshname_part_mapping = {}
            
            self.names_perpart = []
            self.cached_vertices_perpart = []
            self.cached_faces_perpart = []
            self.cached_face_labels_perpart = []
            
            total_count = 0
            for i in range(len(self.names)):
                part_indices = self.get_part_indices(i)
                for part_id in part_indices:
                    self.meshname_part_mapping[total_count] = (self.names[i], i, part_id)
                    self.names_perpart.append(self.names[i])
                    self.cached_vertices_perpart.append(self.cached_vertices[i])
                    self.cached_faces_perpart.append(self.cached_faces[i])
                    self.cached_face_labels_perpart.append(self.cached_face_labels[i])
                    total_count += 1
            # now the dataset is divided into parts
            self.names = self.names_perpart
            self.cached_vertices = self.cached_vertices_perpart
            self.cached_faces = self.cached_faces_perpart
            self.cached_face_labels = self.cached_face_labels_perpart
            
        if self.all_parts_per_epoch:
            print("Number of parts loaded: ", total_count)
            print("Number of unique meshes loaded: ", len(set(self.names_perpart)))
        
        else:
            print(len(self.cached_vertices), "meshes loaded")
            print("Number of unique meshes loaded: ", len(set(self.names)))

    def len(self):
        return len(self.cached_vertices)
    
    def get_part_indices(self, idx):
        face_labels = self.cached_face_labels[idx]
        unique_face_labels = sorted(np.unique(face_labels))
        return unique_face_labels

    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices = scale_vertices(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
        vertices = normalize_vertices(vertices)
        if self.shift_augment:
            vertices = shift_vertices(vertices)
        triangles, normals, areas, angles, vertices, faces = create_feature_stack(vertices, faces, self.num_tokens, self.face_order_augment)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(features[:, :9]).float()
        if self.use_start_stop:
            features = np.concatenate([np.zeros((1, features.shape[1])), features], axis=0)
            target = torch.cat([target, torch.ones(1, 9) * 0.5], dim=0)
            face_neighborhood = face_neighborhood + 1
        if self.only_backward_edges:
            face_neighborhood = face_neighborhood[face_neighborhood[:, 1] > face_neighborhood[:, 0], :]
            # face_neighborhood = modify so that only edges in backward direction are present
        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return features, target, vertices, faces, face_neighborhood

    def get(self, idx):
        features, target, _, _, face_neighborhood = self.get_all_features_for_shape(idx)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long())


class TriangleNodesWithFaces(TriangleNodes):

    def __init__(self, config, split, scale_augment, shift_augment, force_category, face_order_augment):
        super().__init__(config, split, scale_augment, shift_augment, force_category, face_order_augment=face_order_augment)
        
    
    def get_one_ring_for_each_part(self,idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        face_labels = self.cached_face_labels[idx] # per functional uid
        
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices, scale_x, scale_y, scale_z = scale_vertices_returnscale(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
        else:
            scale_x, scale_y, scale_z = 1, 1, 1
        vertices, bounds = normalize_vertices_returnbounds(vertices)
        if self.shift_augment:
            vertices, shift_x, shift_y, shift_z = shift_vertices_returnshifts(vertices)
        else:
            shift_x, shift_y, shift_z = 0, 0, 0
        
        # re-order the vertices, faces and face labels for the WHOLE shape
        vertices, faces, face_labels = sort_vertices_and_faces_wfacelabels(vertices, faces, face_labels, self.num_tokens, self.face_order_augment)
        
        semantic_face_labels = []
        for label in face_labels:
            if label < 100:
                semantic_face_labels.append(label)
            else:
                semantic_face_labels.append(label // 100)
        assert len(semantic_face_labels) == len(face_labels)
        assert len(semantic_face_labels) == len(faces)
        
        
        whole_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False, maintain_order=True)
        face_adjacency = whole_mesh.face_adjacency
        
        # Create a graph where each node is a face and edges connect adjacent faces
        face_graph = nx.Graph()
        face_graph.add_edges_from(face_adjacency)
        
        unique_face_labels = self.get_functional_part_ordering(idx)
        
        junction_uid_mapping = {}
        for i, uid in enumerate(unique_face_labels):
            sem_label = uid if uid < 100 else uid // 100
            if sem_label == 5:
                junction_uid_mapping[uid] = []
            else:
                junction_uid_mapping[uid] = [prev_uid for prev_uid in unique_face_labels[:i]]
        
        one_ring_faces = {}
        one_ring_face_labels = {}
        
        assert np.all(vertices == whole_mesh.vertices)
        assert np.all(faces == whole_mesh.faces)

        for uid in unique_face_labels:
            one_ring_faces[uid] = []
            one_ring_face_labels[uid] = []
            
            junction_face_sem_labels = junction_uid_mapping[uid]
            if len(junction_face_sem_labels) == 0:
                continue
            
            curr_part_face_ids = [i for i, label in enumerate(face_labels) if label == uid]
            
            one_ring_neighbors = set()  # Using a set to avoid duplicates
            one_ring_neighbors_labels = set()

            # For each face in the current part, find its adjacent faces
            for face_idx in curr_part_face_ids:
                if face_idx not in face_graph: # no adjacent faces
                    continue
                adjacent_faces = list(face_graph[face_idx].keys())
                
                # Filter the adjacent faces to include only those from different parts
                for adj_face in adjacent_faces:
                    if face_labels[adj_face] != uid and face_labels[adj_face] in junction_face_sem_labels:
                        one_ring_neighbors.add(adj_face)  # Add to the set to avoid duplicates
                        one_ring_neighbors_labels.add(face_labels[adj_face])
                        
            if len(one_ring_neighbors) >= 10:
                one_ring_faces[uid].extend(one_ring_neighbors)
                one_ring_face_labels[uid].extend(one_ring_neighbors_labels)
            else:
                dist_threshold = 0.015
                # relaxed one-ring: consider faces from other parts that are close to the current part
                current_part_mesh = trimesh.Trimesh(vertices=vertices, faces=[faces[i] for i in curr_part_face_ids], process=True, maintain_order=True)
                max_length_junction_faces = np.maximum(700 - len(current_part_mesh.faces), 10)
                
                query_points = []
                query_face_ids = []
                query_face_labels = []
                for f_idx, f in enumerate(faces):
                    if face_labels[f_idx] != uid and face_labels[f_idx] in junction_face_sem_labels:
                        
                        # Calculate the centroid of the current part
                        current_part_centroid = current_part_mesh.centroid
                        # Get the vertices of the current face
                        face_vertices = vertices[f]
                        # Calculate the maximum z-value of the face
                        max_z = np.max(face_vertices[:, 2])
                        # Only include the face if its maximum z-value is not higher than the centroid's z-value
                        if max_z <= current_part_centroid[2]:
                            query_points.extend(vertices[f])
                            query_face_ids.append(f_idx)
                            query_face_labels.append(face_labels[f_idx])
                query_points = np.array(query_points)
                
                _, dists, _ = trimesh.proximity.closest_point(current_part_mesh, query_points)
                dists_per_face = dists.reshape(-1, 3)
                
                min_dists = np.min(dists_per_face, axis=1)
                
                junction_flags = min_dists < dist_threshold
                
                junction_face_ids = [query_face_ids[i] for i, is_junction in enumerate(junction_flags) if is_junction]
                junction_face_labels = [query_face_labels[i] for i, is_junction in enumerate(junction_flags) if is_junction]
                
                # If no faces meet the distance threshold, get the 10 closest faces
                if len(junction_face_ids) == 0:
                    closest_10_indices = np.argsort(min_dists)[:10]
                    junction_face_ids = [query_face_ids[i] for i in closest_10_indices]
                    junction_face_labels = [query_face_labels[i] for i in closest_10_indices]

                #         junction_face_labels.extend(face_labels[neighbor])
                
                one_ring_faces[uid].extend(junction_face_ids)
                one_ring_face_labels[uid].extend(junction_face_labels)
                
                if len(one_ring_faces[uid]) + len(current_part_mesh.faces) > 700:
                    # Sort junction faces by distance
                    sorted_junction_faces = sorted(zip(junction_face_ids, min_dists), key=lambda x: x[1])
                    # Keep only the closest faces up to the maximum allowed
                    kept_junction_faces = [face_id for face_id, _ in sorted_junction_faces[:max_length_junction_faces]]
                    kept_junction_face_labels = [face_labels[face_id] for face_id in kept_junction_faces]
                    one_ring_faces[uid] = kept_junction_faces
                    one_ring_face_labels[uid] = kept_junction_face_labels
        
        
        junction_indicators = {}
        for uid in unique_face_labels:
            junction_indicators[uid] = np.zeros(len(faces), dtype=int)

        # For each part, mark which of its faces are junction faces for other parts
        for uid in unique_face_labels:
            curr_part_face_ids = set([i for i, label in enumerate(face_labels) if label == uid])
            
            # Check against all other parts
            for other_uid in unique_face_labels:
                if other_uid != uid:
                    # Find which faces of the current part are in the one-ring of the other part
                    junction_faces = set(one_ring_faces[other_uid]) & curr_part_face_ids
                    
                    # Mark these faces in the junction indicators
                    for face_idx in junction_faces:
                        junction_indicators[uid][face_idx] = 1
            
        return one_ring_faces, one_ring_face_labels, junction_indicators, (scale_x, scale_y, scale_z, shift_x, shift_y, shift_z, bounds), vertices, faces, face_labels
    
    def get_functional_part_ordering(self, idx): # for articulated objects
        mesh_name = self.names[idx]
        mesh_category = mesh_name.split("_")[0]
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        face_labels = self.cached_face_labels[idx]
        
        unique_face_labels = sorted(np.unique(face_labels))
        
        functional_order = {
            '02933112': [-1],
            '03001627': [5, 0, -1],
            '04379243': [5, -1]
        }
        
        uid_heights = {}
        for uid in unique_face_labels:
            curr_part_face_ids = [i for i, label in enumerate(face_labels) if label == uid]
            curr_faces = [faces[i] for i in curr_part_face_ids]
            aabb = trimesh.Trimesh(vertices=vertices, faces=curr_faces, process=True).bounding_box
            part_bbox_verts = np.asarray(aabb.vertices)
            part_bbox_faces = aabb.faces.tolist()
            part_bbox_verts, part_bbox_faces = sort_vertices_and_faces_womerge(part_bbox_verts, part_bbox_faces, self.num_tokens, self.face_order_augment)
            uid_heights[uid] = part_bbox_verts[:, 2].min()
            # uid_heights[uid] = np.mean(part_bbox_verts[:, 2])
        
        def custom_sort_key(uid):
            sem_label = uid if uid < 100 else uid // 100
            if mesh_category in functional_order:
                if sem_label in functional_order[mesh_category]:
                    return (functional_order[mesh_category].index(sem_label), uid_heights[uid])
                else:
                    return (len(functional_order[mesh_category]), uid_heights[uid])
            else:
                return uid_heights[uid]
        
        # Sort UIDs based on semantic order and height
        sorted_uids = sorted(unique_face_labels, key=custom_sort_key)
        
        if mesh_category == '04379243':  # Table
            # For table, keep 5 and -1 at the beginning, sort others by height
            base_and_wheel = [uid for uid in sorted_uids if uid == -1 or uid >= 500]
            others = [uid for uid in sorted_uids if uid not in base_and_wheel]
            others_sorted = sorted(others, key=lambda x: uid_heights[x])
            sorted_uids = base_and_wheel + others_sorted
        
        return sorted_uids
    
    def get_part_features(self, idx, vertices, faces, face_labels, junction_faces, junction_face_labels, junction_labels, part_id=None): # v,f,fl are whole shape, scaled/normed/shifted, sorted
        
        if not self.all_parts_per_epoch:
            unique_face_labels = sorted(np.unique(face_labels))
            if part_id is None or part_id not in face_labels:
                part_id = np.random.choice(unique_face_labels)
        else:
            # print('all parts per epoch')
            mesh_name, _, part_id = self.meshname_part_mapping[idx]
            
        junction_indicators = junction_labels[part_id] # for those faces in the current part, which can be junction face for other parts
        
        curr_part_face_ids = [i for i, label in enumerate(face_labels) if label == part_id]
        curr_faces = [faces[i] for i in curr_part_face_ids] # connectivity still based on the whole shape
        curr_junction_indicators = junction_indicators[curr_part_face_ids]
        
        curr_vertices, curr_faces, junction_targets = sort_vertices_and_faces_wfacelabels(vertices[:,[1,2,0]], curr_faces, curr_junction_indicators, self.num_tokens, self.face_order_augment)
        
        junction_needed = junction_faces[part_id] # junction faces ids for the current part
        if len(junction_needed) > 0:
            junction_faces = [faces[i] for i in junction_needed]
            junction_vertices, junction_faces = sort_vertices_and_faces_womerge(vertices[:,[1,2,0]], junction_faces, self.num_tokens, self.face_order_augment)
            junction_existance = True
        else:
            # print('no junction faces for the current part {}'.format(part_id))
            junction_vertices, junction_faces = curr_vertices.copy(), curr_faces.copy()
            junction_existance = False
        junction_triangles = junction_vertices[junction_faces, :]
        junction_triangles, junction_normals, junction_areas, junction_angles = create_feature_stack_from_triangles(junction_triangles)
        junction_features = np.hstack([junction_triangles, junction_normals, junction_areas, junction_angles])
        junction_face_neighborhood = np.array(trimesh.Trimesh(vertices=junction_vertices, faces=junction_faces, process=False).face_neighborhood)  # type: ignore
        
        triangles = curr_vertices[curr_faces, :]
        triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
        
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=curr_vertices, faces=curr_faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(features[:, :9]).float()
        if self.use_start_stop:
            features = np.concatenate([np.zeros((1, features.shape[1])), features], axis=0)
            target = torch.cat([target, torch.ones(1, 9) * 0.5], dim=0)
            face_neighborhood = face_neighborhood + 1
        if self.only_backward_edges:
            face_neighborhood = face_neighborhood[face_neighborhood[:, 1] > face_neighborhood[:, 0], :]
            # face_neighborhood = modify so that only edges in backward direction are present
        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        
        return (features, target, curr_vertices, curr_faces, face_neighborhood, part_id), (junction_features, junction_targets, junction_vertices, junction_faces, junction_face_neighborhood, junction_existance)

    def get(self, idx):
        junction_faces, junction_face_labels, junction_indicators, aug_params, vertices_all, faces_all, face_labels_all = self.get_one_ring_for_each_part(idx) # vertices_all, faces_all, face_labels_all are whole shape, auged, sorted; valid junction only for articulated part
        current_part_features, current_part_junctions = self.get_part_features(idx, vertices_all, faces_all, face_labels_all, junction_faces, junction_face_labels, junction_indicators, part_id=None)
    
        features, target, vertices, faces, face_neighborhood, part_id_actual = current_part_features
        current_part_junction_features, current_part_junction_targets, current_part_junction_vertices, current_part_junction_faces, current_part_junction_face_neighborhood, current_part_junction_existance = current_part_junctions
        
        return GeometricData(x=torch.from_numpy(features).float(), y=target,
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long(), part_id=part_id_actual,
                             junction_targets=torch.tensor(current_part_junction_targets).float())

            
class TriangleNodesWithFacesDataloader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, exclude_keys=None, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)
        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=FaceCollator(follow_batch, exclude_keys),
            **kwargs,
        )


class FaceCollator(GeometricCollator):

    def __init__(self, follow_batch, exclude_keys):
        super().__init__(follow_batch, exclude_keys)

    def __call__(self, batch):
        elem = batch[0]

        num_vertices = 0
        for b_idx in range(len(batch)):
            batch[b_idx].faces += num_vertices
            num_vertices += batch[b_idx].num_vertices

        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # pragma: no cover
        raise NotImplementedError

class TriangleNodesWithSequenceIndices(TriangleNodes):

    vq_depth_factor = 1

    def __init__(self, config, split, scale_augment, shift_augment, force_category):
        super().__init__(config, split, scale_augment=scale_augment, shift_augment=shift_augment, force_category=force_category)
        vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.vq_depth = vq_cfg.embed_levels
        self.block_size = config.block_size
        max_inner_face_len = 0
        self.padding = int(config.padding * self.block_size)
        self.sequence_stride = config.sequence_stride
        for i in range(len(self.cached_vertices)):
            self.cached_vertices[i] = np.array(self.cached_vertices[i])
            for j in range(len(self.cached_faces[i])):
                max_inner_face_len = max(max_inner_face_len, len(self.cached_faces[i][j]))
        print('Longest inner face sequence', max_inner_face_len)
        assert max_inner_face_len == 3, f"Only triangles are supported, but found a face with {max_inner_face_len}."
        self.sequence_indices = []
        max_face_sequence_len = 0
        min_face_sequence_len = 1e7
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            max_face_sequence_len = max(max_face_sequence_len, sequence_len)
            min_face_sequence_len = min(min_face_sequence_len, sequence_len)
            self.sequence_indices.append((i, 0, False))
            # for j in range(config.sequence_stride, max(1, sequence_len - self.block_size + self.padding + 1), config.sequence_stride):  # todo: possible bug? +1 added recently
            #     self.sequence_indices.append((i, j, True if split == 'train' else False))
            # if sequence_len > self.block_size: 
            #     self.sequence_indices.append((i, sequence_len - self.block_size, False))
        print('Length of', split, len(self.sequence_indices))
        print('Shortest face sequence', min_face_sequence_len)
        print('Longest face sequence', max_face_sequence_len)

    def len(self):
        return len(self.sequence_indices)
    
    def get(self, idx):
        i, j, randomness = self.sequence_indices[idx]
        if randomness:
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        features, target, _, _, face_neighborhood = self.get_all_features_for_shape(i)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long(), js=torch.tensor(j).long())
    
    def plot_sequence_lenght_stats(self):
        sequence_lengths = []
        for i in range(len(self.cached_faces)):
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            sequence_lengths.append(sequence_len)
        import matplotlib.pyplot as plt
        plt.hist(sequence_lengths, bins=32)
        plt.ylim(0, 100)
        plt.show()
        return sequence_lengths


class TriangleNodesWithFacesAndSequenceIndices(TriangleNodesWithSequenceIndices):
    vq_depth_factor = 3
    def __init__(self, config, split, scale_augment, shift_augment, force_category):
        super().__init__(config, split, scale_augment, shift_augment, force_category)

    def get(self, idx):
        i, j, randomness = self.sequence_indices[idx]
        if randomness:
            sequence_len = len(self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        features, target, vertices, faces, face_neighborhood = self.get_all_features_for_shape(i)
        return GeometricData(x=torch.from_numpy(features).float(),
                             y=target, mesh_name=self.names[i], edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             js=torch.tensor(j).long(), num_vertices=vertices.shape[0],
                             faces=torch.from_numpy(np.array(faces)).long())


class Triangles(Dataset):

    def __init__(self, config, split, scale_augment, shift_augment):
        super().__init__()
        data_path = Path(config.dataset_root)
        self.cached_vertices = []
        self.cached_faces = []
        self.names = []
        self.scale_augment = scale_augment
        self.shift_augment = shift_augment
        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
            else:
                multiplier = 1 if split == 'val' else 500
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier

        print(len(self.cached_vertices), "meshes loaded")
        self.features = None
        self.setup_triangles_for_epoch()

    def __len__(self):
        return self.features.shape[0]

    def setup_triangles_for_epoch(self):
        all_features = []
        for idx in tqdm(range(len(self.cached_vertices)), desc="refresh augs"):
            vertices = self.cached_vertices[idx]
            faces = self.cached_faces[idx]
            if self.scale_augment:
                vertices = scale_vertices(vertices)
            vertices = normalize_vertices(vertices)
            if self.shift_augment:
                vertices = shift_vertices(vertices)
            all_features.append(create_feature_stack(vertices, faces)[0])
        self.features = np.vstack(all_features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.features[idx, :9]
        }

    def get_all_features_for_shape(self, idx):
        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        feature_stack = create_feature_stack(vertices, faces)[0]
        return torch.from_numpy(feature_stack).float(), torch.from_numpy(feature_stack[:, :9]).float()


def normal(triangles):
    # The cross product of two sides is a normal vector
    if torch.is_tensor(triangles):
        return torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    else:
        return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1)


def area(triangles):
    # The norm of the cross product of two sides is twice the area
    if torch.is_tensor(triangles):
        return torch.norm(normal(triangles), dim=1) / 2
    else:
        return np.linalg.norm(normal(triangles), axis=1) / 2


def angle(triangles):
    v_01 = triangles[:, 1] - triangles[:, 0]
    v_02 = triangles[:, 2] - triangles[:, 0]
    v_10 = -v_01
    v_12 = triangles[:, 2] - triangles[:, 1]
    v_20 = -v_02
    v_21 = -v_12
    if torch.is_tensor(triangles):
        return torch.stack([angle_between(v_01, v_02), angle_between(v_10, v_12), angle_between(v_20, v_21)], dim=1)
    else:
        return np.stack([angle_between(v_01, v_02), angle_between(v_10, v_12), angle_between(v_20, v_21)], axis=1)


def angle_between(v0, v1):
    v0_u = unit_vector(v0)
    v1_u = unit_vector(v1)
    if torch.is_tensor(v0):
        return torch.arccos(torch.clip(torch.einsum('ij,ij->i', v0_u, v1_u), -1.0, 1.0))
    else:
        return np.arccos(np.clip(np.einsum('ij,ij->i', v0_u, v1_u), -1.0, 1.0))


def unit_vector(vector):
    if torch.is_tensor(vector):
        return vector / (torch.norm(vector, dim=-1)[:, None] + 1e-8)
    else:
        return vector / (np.linalg.norm(vector, axis=-1)[:, None] + 1e-8)


def create_feature_stack(vertices, faces, num_tokens, face_order_augment):
    vertices, faces = sort_vertices_and_faces(vertices, faces, num_tokens, face_order_augment)
    # need more features: positions, angles, area, cross_product
    triangles = vertices[faces, :]
    triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
    return triangles, normals, areas, angles, vertices, faces


def create_feature_stack_from_triangles(triangles):
    t_areas = area(triangles) * 1e3
    t_angles = angle(triangles) / float(np.pi)
    t_normals = unit_vector(normal(triangles))
    return triangles.reshape(-1, 9), t_normals.reshape(-1, 3), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)