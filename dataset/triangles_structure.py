import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import pickle
import torch.utils.data
from pathlib import Path
from torch.utils.data import Dataset, default_collate
import trimesh
import numpy as np
import torch
import omegaconf
from typing import Mapping, Sequence

from torch_geometric.loader.dataloader import Collater as GeometricCollator
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import Dataset as GeometricDataset, Batch
from util.misc import normalize_vertices, scale_vertices, shift_vertices, scale_vertices_returnscale, shift_vertices_returnshifts, normalize_vertices_returnbounds
from dataset.quantize_and_tokenize_soup import sort_vertices_and_faces, quantize_coordinates, sort_vertices_and_faces_womerge
from tqdm import tqdm
from torch_geometric.data.data import BaseData
import json
from scipy.spatial.transform import Rotation as R

class TriangleNodes(GeometricDataset):

    def __init__(self, config, split, scale_augment, shift_augment, force_category=None, use_start_stop=False, only_backward_edges=False, face_order_augment=False, joint_augment=False):
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
        
        self.move_joint_augment = joint_augment

        self.cached_face_labels = []
        self.cached_part_features = []

        self.cached_contact_infos = []
        self.cached_face_rings = []
        self.cached_face_label_rings = []
        self.cached_pointclouds_train = []
        self.cached_pointcloudlabels_train = []
        self.cached_pointclouds_val = []
        self.cached_pointcloudlabels_val = []
        
        with open(config.load_articulation_info, 'rb') as fptr:
            self.articulation_info = json.load(fptr)
            
        with open(config.load_structure_info, 'rb') as fptr:
            self.structure_info = json.load(fptr)

        with open(config.load_part_class_text_feature, 'rb') as fptr:
            text_features_ = pickle.load(fptr)
            self.text_features_partclass = text_features_
            
        with open(config.load_class_feature_mapping, 'rb') as fptr:
            self.semantic_feature_decipher = pickle.load(fptr)
            
        with open(config.load_joint_text_feature, 'rb') as fptr:
            text_features_ = pickle.load(fptr)
            self.text_features_jointtype = text_features_
            
        # construct a array for semantic retrieval
        self.semantic_retrieval_array = []
        for class_id in self.text_features_partclass:
            class_sem_dict = self.text_features_partclass[class_id]
            for part_label in class_sem_dict:
                class_features = class_sem_dict[part_label]
                self.semantic_retrieval_array.append(class_features)
        self.semantic_retrieval_array = np.stack(self.semantic_retrieval_array, axis=0)
        
        self.joint_types = {'fixed': 0, 'revolute': 1, 'prismatic': 2}

        with open(data_path, 'rb') as fptr:
            data = pickle.load(fptr)
            if force_category is not None:
                for s in ['train', 'val']:
                    data[f'vertices_{s}'] = [data[f'vertices_{s}'][i] for i in range(len(data[f'vertices_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'faces_{s}'] = [data[f'faces_{s}'][i] for i in range(len(data[f'faces_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'pointclouds_{s}'] = [data[f'pointclouds_{s}'][i] for i in range(len(data[f'pointclouds_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'pointcloudlabels_{s}'] = [data[f'pointcloudlabels_{s}'][i] for i in range(len(data[f'pointcloudlabels_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                    data[f'name_{s}'] = [data[f'name_{s}'][i] for i in range(len(data[f'name_{s}'])) if data[f'name_{s}'][i].split('_')[0] == force_category]
                if len(data[f'vertices_val']) == 0:
                    data[f'vertices_val'] = data[f'vertices_train']
                    data[f'faces_val'] = data[f'faces_train']
                    data[f'pointclouds_val'] = data[f'pointclouds_train']
                    data[f'pointcloudlabels_val'] = data[f'pointcloudlabels_train']
                    data[f'name_val'] = data[f'name_train']
            if not config.overfit:
                self.names = data[f'name_{split}']
                self.cached_vertices = data[f'vertices_{split}']
                self.cached_faces = data[f'faces_{split}']
                self.cached_pointclouds = data[f'pointclouds_{split}']
                self.cached_pointcloudlabels = data[f'pointcloudlabels_{split}']
                
                val_names_set = set([name.split("_")[1] for name in data['name_val']])
                indices = []
                for i, train_name in enumerate(self.names):
                    object_id = train_name.split("_")[1]
                    if split != 'train' or object_id not in val_names_set:
                        indices.append(i)
            else:
                multiplier = 16 if split == 'val' else 256
                self.names = data[f'name_train'][:1] * multiplier
                self.cached_vertices = data[f'vertices_train'][:1] * multiplier
                self.cached_faces = data[f'faces_train'][:1] * multiplier
                self.cached_pointclouds = data[f'pointclouds_train'][:1] * multiplier
                self.cached_pointcloudlabels = data[f'pointcloudlabels_train'][:1] * multiplier
                        
        self.load_geometry_features = config.load_geometry_features
        if self.load_geometry_features:
            geometry_feature_path = Path(config.geometry_feature_path)
            with open(geometry_feature_path, 'rb') as fp:
                self.geometry_features = pickle.load(fp)
        
        print(len(self.cached_vertices), "meshes loaded")
        print("Number of unique meshes loaded: ", len(set(self.names)))

    def len(self):
        return len(self.cached_vertices)

    def get_all_features_for_shape(self, idx):

        vertices = self.cached_vertices[idx]
        faces = self.cached_faces[idx]
        face_labels = self.cached_face_labels[idx]
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

        unique_face_labels = sorted(np.unique(face_labels))
        if 4 in unique_face_labels:
            unique_face_labels.remove(4)

        corners_whole_shape = []
        faces_bboxs = []
        face_labels_bbox = []

        semantic_label_features = []

        geometry_features = []

        for uid in unique_face_labels:
            curr_part_face_ids = [i for i, label in enumerate(face_labels) if label == uid]
            # connectivity still based on the whole shape
            curr_faces = [faces[i] for i in curr_part_face_ids]

            # get aabb of the part
            aabb = trimesh.Trimesh(vertices=vertices, faces=curr_faces, process=True).bounding_box

            # bbox as triangle soup
            part_bbox_verts = np.asarray(aabb.vertices)
            if self.scale_augment:
                x_lims_bbox = (0.9, 1.1)
                y_lims_bbox = (0.9, 1.1)
                z_lims_bbox = (0.9, 1.1)
                part_bbox_verts = scale_vertices(part_bbox_verts, x_lims=x_lims_bbox, y_lims=y_lims_bbox, z_lims=z_lims_bbox)

            part_bbox_faces = aabb.faces.tolist()
            part_bbox_verts, part_bbox_faces = sort_vertices_and_faces(part_bbox_verts, part_bbox_faces, self.num_tokens, self.face_order_augment)
            bbox_faces_ = [(np.asarray(f)+len(corners_whole_shape)).tolist() for f in part_bbox_faces]

            # treat bbox as triangle soup
            corners_whole_shape.extend(part_bbox_verts)
            faces_bboxs.extend(bbox_faces_)
            face_labels_bbox.extend([uid] * len(part_bbox_faces))


            text_features = torch.from_numpy(self.text_features_partclass[uid])
            text_features = torch.repeat_interleave(text_features.unsqueeze(0), len(part_bbox_faces), dim=0)
            semantic_label_features.extend(text_features)

            part_abstraction = self.cached_part_features[idx][uid][:128].reshape(1, -1)
            part_abstraction = torch.repeat_interleave(torch.from_numpy(part_abstraction), len(part_bbox_faces), dim=0)  # (num_faces, 128)
            geometry_features.append(part_abstraction)
            
        
        vertices = np.array(corners_whole_shape)
        faces = faces_bboxs

        # treat the whole shape as triangle soup
        triangles = vertices[faces, :]
        triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(features[:, :9]).float()

        semantic_label_features = torch.stack(semantic_label_features)  # (num_faces(36/48), 768)
        target_semantic = torch.from_numpy(np.asarray(face_labels_bbox)).long()  # (num_faces(36/48))

        geometry_features = torch.cat(geometry_features, dim=0)  # (num_faces, 128)

        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return features, target, vertices, faces, face_neighborhood, semantic_label_features, target_semantic, geometry_features

    def get_structure_ordering(self, idx):
        mesh_name = self.names[idx]
        mesh_category = mesh_name.split("_")[0]
        vertices = self.cached_pointclouds[idx]
        face_labels = self.cached_pointcloudlabels[idx]
        
        unique_face_labels = sorted(np.unique(face_labels))
        
        # Define semantic ordering for each category
        semantic_order = {
            '02933112': [0, 1],
            '03001627': [5, 4, 0, 1, 2, 3], 
            '04379243': [5, 0, 4]
        }
        
        uid_heights = {}
        for uid in unique_face_labels:
            curr_points = vertices[face_labels == uid]
            if curr_points.shape[0] % 3 != 0:
                num_dummy_points = 3 - curr_points.shape[0] % 3
                dummy_points = curr_points[np.random.choice(curr_points.shape[0], num_dummy_points)]
                curr_points = np.concatenate([curr_points, dummy_points], axis=0)
            curr_faces = np.array(list(range(curr_points.shape[0]))).reshape(-1, 3)

            # get aabb of the part
            aabb = trimesh.Trimesh(vertices=curr_points, faces=curr_faces, process=True).bounding_box
            part_bbox_verts = np.asarray(aabb.vertices)
            part_bbox_faces = aabb.faces.tolist()
            part_bbox_verts, part_bbox_faces = sort_vertices_and_faces_womerge(part_bbox_verts, part_bbox_faces, self.num_tokens, self.face_order_augment)
            uid_heights[uid] = part_bbox_verts[:, 2].min()
            # uid_heights[uid] = np.mean(part_bbox_verts[:, 2])
        
        def custom_sort_key(uid):
            sem_label = uid if uid < 100 else uid // 100
            if mesh_category in semantic_order:
                if sem_label in semantic_order[mesh_category]:
                    return (semantic_order[mesh_category].index(sem_label), uid_heights[uid])
                else:
                    return (len(semantic_order[mesh_category]), uid_heights[uid])
            else:
                return uid_heights[uid]
        
        # Sort UIDs based on semantic order and height
        sorted_uids = sorted(unique_face_labels, key=custom_sort_key)
        
        if mesh_category == '02933112':  # Cabinet
            # For cabinet, keep 0 and 1 at the beginning, then sort others by height
            base_and_frame = [uid for uid in sorted_uids if uid < 200]
            others = [uid for uid in sorted_uids if uid >= 200]
            others_sorted = sorted(others, key=lambda x: uid_heights[x])
            sorted_uids = base_and_frame + others_sorted
        elif mesh_category == '04379243':  # Table
            # For table, keep 5 and 0 at the beginning, 4 at the end, sort others by height
            base_and_wheel = [uid for uid in sorted_uids if uid ==0 or uid >=500]
            # drawers = [uid for uid in sorted_uids if 200<=uid<300]
            table_top = [uid for uid in sorted_uids if uid == 4]
            others = [uid for uid in sorted_uids if uid not in base_and_wheel and uid not in table_top]
            others_sorted = sorted(others, key=lambda x: uid_heights[x])
            sorted_uids = base_and_wheel + others_sorted + table_top
        
        return sorted_uids

    def get_all_features_for_shape_pc(self, idx):
        mesh_name = self.names[idx]
        obj_cat, obj_id = mesh_name.split("_")[0], mesh_name.split("_")[1]
        
        vertices = self.cached_pointclouds[idx]
        face_labels = self.cached_pointcloudlabels[idx]
        
        obj_partnetid = obj_id
        
        articulation_exits = False
        if obj_partnetid in self.articulation_info:
            articulation_exits = True
            obj_articulation_info = self.articulation_info[obj_partnetid]["articulation_info"]
        
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices, x_scale, y_scale, z_scale = scale_vertices_returnscale(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
        vertices, bounds = normalize_vertices_returnbounds(vertices)
        if self.shift_augment:
            vertices, x_shift, y_shift, z_shift = shift_vertices_returnshifts(vertices)
            
        # new_face_labels = [] # on points
        new_face_labels_bbox = [] # on faces
        class_labels = {}
        class_inst_label_mapping = {}

        for uid in face_labels:
            if uid >= 100:
                class_id, instance_id = uid // 100, uid % 100
                class_inst_label_mapping[uid] = "{}_{}".format(class_id, instance_id)
                class_labels[uid] = class_id
            else:
                class_labels[uid] = uid
                class_inst_label_mapping[uid] = "{}_0".format(uid)

        unique_face_labels = self.get_structure_ordering(idx)

        corners_whole_shape = []
        faces_bboxs = []
        face_labels_bbox = []

        semantic_label_features = []
        geometry_features = []

        articulation_existance_labels = []
        joint_locations = []
        joint_orientations = []
        joint_types = []
        joint_limit_as = []
        joint_limit_bs = []
        
        # for vis
        joint_locs = []
        joint_oris = []
        joint_types_vis = []
        
        joint_type_labels = []
            
        for uid in unique_face_labels:
            curr_points = vertices[face_labels == uid]
            semantic_class = class_labels[uid] # class-semantic label

            part_geometry_features = torch.from_numpy(self.geometry_features[obj_id][uid])
            
            if curr_points.shape[0] % 3 != 0:
                num_dummy_points = 3 - curr_points.shape[0] % 3
                dummy_points = curr_points[np.random.choice(curr_points.shape[0], num_dummy_points)]
                curr_points = np.concatenate([curr_points, dummy_points], axis=0)
            curr_faces = np.array(list(range(curr_points.shape[0]))).reshape(-1, 3)

            # get aabb of the part
            aabb = trimesh.Trimesh(vertices=curr_points, faces=curr_faces, process=True).bounding_box
            
            # if the extent along one direction is very smale, scale it to at least 0.1
            extents = aabb.extents
            min_extent = 0.03
            adjusted_extents = np.maximum(extents, min_extent)
            if not np.array_equal(extents, adjusted_extents):
                # Center of the current bounding box
                center = aabb.centroid
                new_bbox = trimesh.primitives.Box(extents=adjusted_extents, transform=trimesh.transformations.translation_matrix(center))
            else:
                new_bbox = aabb

            # bbox as triangle soup
            part_bbox_verts = np.asarray(new_bbox.vertices)
            
            part_bbox_faces = new_bbox.faces.tolist()

            part_bbox_verts, part_bbox_faces = sort_vertices_and_faces_womerge(part_bbox_verts, part_bbox_faces, self.num_tokens, self.face_order_augment)

            bbox_faces_ = [(np.asarray(f)+len(corners_whole_shape)).tolist() for f in part_bbox_faces]
            
            part_geometry_features = part_geometry_features.unsqueeze(0).repeat(len(part_bbox_faces), 1)
            geometry_features.append(part_geometry_features)
            
            faces_bboxs.extend(bbox_faces_)
            
            face_labels_bbox.extend([semantic_class] * len(part_bbox_faces))
            new_face_labels_bbox.extend([uid] * len(part_bbox_faces))
            
            text_features = torch.from_numpy(self.text_features_partclass[obj_cat][semantic_class])
            text_features = torch.repeat_interleave(text_features.unsqueeze(0), len(part_bbox_faces), dim=0)
            semantic_label_features.extend(text_features)
            
            
            if articulation_exits: # in the whole shape, there is articulation
                find_match = False
                if uid in class_inst_label_mapping:
                    class_instance_id = class_inst_label_mapping[uid]

                    # check whether the part_id or part_name is in the articulation part list
                    for articulated_part in obj_articulation_info:
                        if str(articulated_part['matched_class_instance']) == str(class_instance_id):
                            find_match = True
                            j_location = np.array(articulated_part["canonicalized"]["origin"]).reshape(1, -1)
                            j_orientation = np.array(articulated_part["canonicalized"]["direction"]).reshape(1, -1)
                            j_type = articulated_part["joint"]
                            
                            # NOTE: the actual value of the joint limit should be double-checked, as CAGE multiplies with 0.6
                            if j_type == "revolute":
                                j_limit_b = torch.Tensor([articulated_part["raw"]["limit"]["b"]*np.pi/180, articulated_part["raw"]["limit"]["b"]*np.pi/180, articulated_part["raw"]["limit"]["b"]*np.pi/180]).reshape(1, -1).float()
                            elif j_type == "prismatic":
                                j_limit_b = torch.Tensor((articulated_part["raw"]["limit"]["b"], articulated_part["raw"]["limit"]["b"], articulated_part["raw"]["limit"]["b"])).reshape(1, -1).float()
                            else:
                                raise ValueError("Unknown joint type")
                            
                            if j_type == "revolute":
                                if self.scale_augment:
                                    j_location = np.stack([j_location[:, 0] * x_scale, j_location[:, 1] * y_scale, j_location[:, 2] * z_scale], axis=-1)
                                
                                # transformed to the location within the normalized vertices 
                                j_location = j_location - (bounds[0] + bounds[1])[None, :] / 2
                                j_location = j_location / (bounds[1] - bounds[0]).max()
                                if self.shift_augment:
                                    j_location = np.stack([j_location[:, 0] + x_shift, j_location[:, 1] + y_shift, j_location[:, 2] + z_shift], axis=-1)
                    
                            joint_location_ = torch.from_numpy(swap_axis(j_location, self.num_tokens, clip=True))
                            
                            if j_type == "revolute":
                                joint_orientation_ = torch.from_numpy(swap_axis_orientation(j_orientation))
                            else:
                                joint_orientation_ = torch.from_numpy(-swap_axis_orientation(j_orientation))
                                
                            # according to the joint type and limit, augment the bbox locations
                            if self.move_joint_augment:
                                aug_prob = 0.5
                                if j_type == "revolute":
                                    if np.random.rand() < aug_prob:
                                        random_rotation_angle = - 0.5 * np.random.rand() * j_limit_b.mean().item()
                                        part_bbox_verts = rotate_bbox(part_bbox_verts, joint_location_.detach().numpy().reshape(-1), joint_orientation_.detach().numpy().reshape(-1), rotation_angle=random_rotation_angle)
                                elif j_type == "prismatic":
                                    if np.random.rand() < aug_prob:
                                        random_shift = 0.5 * np.random.rand() * j_limit_b.mean().item()
                                        part_bbox_verts += random_shift * joint_orientation_.detach().numpy()
                            
                if find_match:
                    articulation_existance_labels.extend([1] * len(part_bbox_faces))
                    
                else:
                    articulation_existance_labels.extend([0] * len(part_bbox_faces))
                    joint_location_ = torch.zeros(3).reshape(1, -1).float()
                    joint_orientation_ = torch.zeros(3).reshape(1, -1).float()
                    j_type = "fixed"
                    j_limit_b = torch.zeros(3).reshape(1, -1).float()
            else:
                articulation_existance_labels.extend([0] * len(part_bbox_faces))
                joint_location_ = torch.zeros(3).reshape(1, -1).float()
                joint_orientation_ = torch.zeros(3).reshape(1, -1).float()
                j_type = "fixed"
                j_limit_b = torch.zeros(3).reshape(1, -1).float()
            
            
            # treat bbox as triangle soup: moved here due to the potential augmentation
            corners_whole_shape.extend(part_bbox_verts)
            
            joint_locs.append(joint_location_.detach().numpy())
            joint_oris.append(joint_orientation_.detach().numpy())
            joint_types_vis.append(j_type)

            text_features_joint_ = torch.from_numpy(self.text_features_jointtype[j_type])
            text_features_joint_ = torch.repeat_interleave(text_features_joint_.unsqueeze(0), len(part_bbox_faces), dim=0)
            
            joint_type_labels.extend(self.joint_types[j_type] * np.ones(len(part_bbox_faces)))
            
            joint_location_ = torch.repeat_interleave(joint_location_, len(part_bbox_faces), dim=0)
            joint_orientation_ = torch.repeat_interleave(joint_orientation_, len(part_bbox_faces), dim=0)
            joint_limit_b_ = torch.repeat_interleave(j_limit_b, len(part_bbox_faces), dim=0)
            
            joint_locations.append(joint_location_)
            joint_orientations.append(joint_orientation_)
            joint_types.append(text_features_joint_)
            joint_limit_bs.append(joint_limit_b_)
            
        
        vertices = np.array(corners_whole_shape)
        faces = faces_bboxs

        # treat the whole shape as triangle soup
        triangles = vertices[faces, :]
        triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(features[:, :9]).float()

        semantic_label_features = torch.stack(semantic_label_features)  # (num_faces, 768)
        target_semantic = torch.from_numpy(np.asarray(face_labels_bbox)).long()  # (num_faces)
        
        geometry_features = torch.cat(geometry_features, dim=0) # (num_faces, 768) / 128
        
        joint_locations = torch.cat(joint_locations, dim=0) # (num_faces, 3)
        joint_orientations = torch.cat(joint_orientations, dim=0) # (num_faces, 3)
        joint_limit_bs = torch.cat(joint_limit_bs, dim=0) # (num_faces, 3)
        joint_types = torch.cat(joint_types, dim=0) # (num_faces, 768)
        
        articulation_features = torch.cat([joint_locations, joint_orientations, joint_limit_bs, joint_types], dim=1).to(torch.float32) # (num_faces, 777)
        target_articulation_existance_labels = torch.from_numpy(np.asarray(articulation_existance_labels)).long() # (num_faces)
        target_articulation_joint_types_labels = torch.from_numpy(np.asarray(joint_type_labels)).long() # (num_faces)

        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return features, target, vertices, faces, face_neighborhood, semantic_label_features, target_semantic, articulation_features, target_articulation_existance_labels, target_articulation_joint_types_labels, geometry_features
    
    def get_all_features_for_part_pc(self, idx, part_id):
        
        mesh_name = self.names[idx]
        
        obj_cat, obj_id = mesh_name.split("_")[0], mesh_name.split("_")[1]
        
        vertices = self.cached_pointclouds[idx]
        face_labels = self.cached_pointcloudlabels[idx]
        
        obj_partnetid = obj_id
        
        articulation_exits = False
        if obj_partnetid in self.articulation_info:
            articulation_exits = True
            obj_articulation_info = self.articulation_info[obj_partnetid]["articulation_info"]
        
        if self.scale_augment:
            if self.low_augment:
                x_lims = (0.9, 1.1)
                y_lims = (0.9, 1.1)
                z_lims = (0.9, 1.1)
            else:
                x_lims = (0.75, 1.25)
                y_lims = (0.75, 1.25)
                z_lims = (0.75, 1.25)
            vertices, x_scale, y_scale, z_scale = scale_vertices_returnscale(vertices, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)
        vertices, bounds = normalize_vertices_returnbounds(vertices)
        if self.shift_augment:
            vertices, x_shift, y_shift, z_shift = shift_vertices_returnshifts(vertices)
            
        new_face_labels_bbox = [] # on faces
        class_labels = {}
        class_inst_label_mapping = {}

        for uid in face_labels:
            if uid >= 100:
                class_id, instance_id = uid // 100, uid % 100
                class_inst_label_mapping[uid] = "{}_{}".format(class_id, instance_id)
                class_labels[uid] = class_id
            else:
                class_labels[uid] = uid
                class_inst_label_mapping[uid] = "{}_0".format(uid)

        unique_face_labels = self.get_structure_ordering(idx)

        corners_whole_shape = []
        faces_bboxs = []
        face_labels_bbox = []

        semantic_label_features = []
        geometry_features = []

        articulation_existance_labels = []
        joint_locations = []
        joint_orientations = []
        joint_types = []
        joint_limit_as = []
        joint_limit_bs = []
        
        # for vis
        joint_locs = []
        joint_oris = []
        joint_types_vis = []
        
        joint_type_labels = []
            
        for uid in unique_face_labels:
            
            if uid != part_id:
                continue
            
            curr_points = vertices[face_labels == uid]
            semantic_class = class_labels[uid] # class-semantic label

            part_geometry_features = torch.from_numpy(self.geometry_features[obj_id][uid])
            
            if curr_points.shape[0] % 3 != 0:
                num_dummy_points = 3 - curr_points.shape[0] % 3
                dummy_points = curr_points[np.random.choice(curr_points.shape[0], num_dummy_points)]
                curr_points = np.concatenate([curr_points, dummy_points], axis=0)
            curr_faces = np.array(list(range(curr_points.shape[0]))).reshape(-1, 3)

            # get aabb of the part
            aabb = trimesh.Trimesh(vertices=curr_points, faces=curr_faces, process=True).bounding_box
            
            # if the extent along one direction is very smale, scale it to at least 0.1
            extents = aabb.extents
            min_extent = 0.03
            adjusted_extents = np.maximum(extents, min_extent)
            if not np.array_equal(extents, adjusted_extents):
                # Center of the current bounding box
                center = aabb.centroid
                new_bbox = trimesh.primitives.Box(extents=adjusted_extents, transform=trimesh.transformations.translation_matrix(center))
            else:
                new_bbox = aabb

            # bbox as triangle soup
            part_bbox_verts = np.asarray(new_bbox.vertices)
            
            part_bbox_faces = new_bbox.faces.tolist()

            part_bbox_verts, part_bbox_faces = sort_vertices_and_faces_womerge(part_bbox_verts, part_bbox_faces, self.num_tokens, self.face_order_augment)

            bbox_faces_ = [(np.asarray(f)+len(corners_whole_shape)).tolist() for f in part_bbox_faces]
            
            
            part_geometry_features = part_geometry_features.unsqueeze(0).repeat(len(part_bbox_faces), 1)
            geometry_features.append(part_geometry_features)

            
            faces_bboxs.extend(bbox_faces_)
            
            face_labels_bbox.extend([semantic_class] * len(part_bbox_faces))
            new_face_labels_bbox.extend([uid] * len(part_bbox_faces))
            
            text_features = torch.from_numpy(self.text_features_partclass[obj_cat][semantic_class])
            text_features = torch.repeat_interleave(text_features.unsqueeze(0), len(part_bbox_faces), dim=0)
            semantic_label_features.extend(text_features)
            
            
            if articulation_exits: # in the whole shape, there is articulation
                find_match = False
                if uid in class_inst_label_mapping:
                    class_instance_id = class_inst_label_mapping[uid]

                    # check whether the part_id or part_name is in the articulation part list
                    for articulated_part in obj_articulation_info:
                        if str(articulated_part['matched_class_instance']) == str(class_instance_id):
                            find_match = True
                            j_location = np.array(articulated_part["canonicalized"]["origin"]).reshape(1, -1)
                            j_orientation = np.array(articulated_part["canonicalized"]["direction"]).reshape(1, -1)
                            j_type = articulated_part["joint"]
                            
                            # NOTE: the actual value of the joint limit should be double-checked, as CAGE multiplies with 0.6
                            if j_type == "revolute":
                                j_limit_b = torch.Tensor([articulated_part["raw"]["limit"]["b"]*np.pi/180, articulated_part["raw"]["limit"]["b"]*np.pi/180, articulated_part["raw"]["limit"]["b"]*np.pi/180]).reshape(1, -1).float()
                            elif j_type == "prismatic":
                                j_limit_b = torch.Tensor((articulated_part["raw"]["limit"]["b"], articulated_part["raw"]["limit"]["b"], articulated_part["raw"]["limit"]["b"])).reshape(1, -1).float()
                            else:
                                raise ValueError("Unknown joint type")
                            
                            if j_type == "revolute":
                                if self.scale_augment:
                                    j_location = np.stack([j_location[:, 0] * x_scale, j_location[:, 1] * y_scale, j_location[:, 2] * z_scale], axis=-1)
                                
                                # transformed to the location within the normalized vertices 
                                j_location = j_location - (bounds[0] + bounds[1])[None, :] / 2
                                j_location = j_location / (bounds[1] - bounds[0]).max()
                                if self.shift_augment:
                                    j_location = np.stack([j_location[:, 0] + x_shift, j_location[:, 1] + y_shift, j_location[:, 2] + z_shift], axis=-1)
                    
                            joint_location_ = torch.from_numpy(swap_axis(j_location, self.num_tokens, clip=True))
                            
                            if j_type == "revolute":
                                joint_orientation_ = torch.from_numpy(swap_axis_orientation(j_orientation))
                            else:
                                joint_orientation_ = torch.from_numpy(-swap_axis_orientation(j_orientation))
                                
                            # according to the joint type and limit, augment the bbox locations
                            if self.move_joint_augment:
                                aug_prob = 0.5
                                if j_type == "revolute":
                                    if np.random.rand() < aug_prob:
                                        random_rotation_angle = - 0.5 * np.random.rand() * j_limit_b.mean().item()
                                        part_bbox_verts = rotate_bbox(part_bbox_verts, joint_location_.detach().numpy().reshape(-1), joint_orientation_.detach().numpy().reshape(-1), rotation_angle=random_rotation_angle)
                                elif j_type == "prismatic":
                                    if np.random.rand() < aug_prob:
                                        random_shift = 0.5 * np.random.rand() * j_limit_b.mean().item()
                                        part_bbox_verts += random_shift * joint_orientation_.detach().numpy()
                            
                if find_match:
                    articulation_existance_labels.extend([1] * len(part_bbox_faces))
                    
                else:
                    articulation_existance_labels.extend([0] * len(part_bbox_faces))
                    joint_location_ = torch.zeros(3).reshape(1, -1).float()
                    joint_orientation_ = torch.zeros(3).reshape(1, -1).float()
                    j_type = "fixed"
                    j_limit_b = torch.zeros(3).reshape(1, -1).float()
            else:
                articulation_existance_labels.extend([0] * len(part_bbox_faces))
                joint_location_ = torch.zeros(3).reshape(1, -1).float()
                joint_orientation_ = torch.zeros(3).reshape(1, -1).float()
                j_type = "fixed"
                j_limit_b = torch.zeros(3).reshape(1, -1).float()
            
            
            # treat bbox as triangle soup: moved here due to the potential augmentation
            corners_whole_shape.extend(part_bbox_verts)
            
            joint_locs.append(joint_location_.detach().numpy())
            joint_oris.append(joint_orientation_.detach().numpy())
            joint_types_vis.append(j_type)

            text_features_joint_ = torch.from_numpy(self.text_features_jointtype[j_type])
            text_features_joint_ = torch.repeat_interleave(text_features_joint_.unsqueeze(0), len(part_bbox_faces), dim=0)
            
            joint_type_labels.extend(self.joint_types[j_type] * np.ones(len(part_bbox_faces)))
            
            joint_location_ = torch.repeat_interleave(joint_location_, len(part_bbox_faces), dim=0)
            joint_orientation_ = torch.repeat_interleave(joint_orientation_, len(part_bbox_faces), dim=0)
            joint_limit_b_ = torch.repeat_interleave(j_limit_b, len(part_bbox_faces), dim=0)
            
            joint_locations.append(joint_location_)
            joint_orientations.append(joint_orientation_)
            joint_types.append(text_features_joint_)
            joint_limit_bs.append(joint_limit_b_)
            
        
        vertices = np.array(corners_whole_shape)
        faces = faces_bboxs
        
        # treat the whole shape as triangle soup
        triangles = vertices[faces, :]
        triangles, normals, areas, angles = create_feature_stack_from_triangles(triangles)
        features = np.hstack([triangles, normals, areas, angles])
        face_neighborhood = np.array(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).face_neighborhood)  # type: ignore
        target = torch.from_numpy(features[:, :9]).float()

        semantic_label_features = torch.stack(semantic_label_features)  # (num_faces, 768)
        target_semantic = torch.from_numpy(np.asarray(face_labels_bbox)).long()  # (num_faces)
        
        geometry_features = torch.cat(geometry_features, dim=0) # (num_faces, 768) / 128
        
        joint_locations = torch.cat(joint_locations, dim=0) # (num_faces, 3)
        joint_orientations = torch.cat(joint_orientations, dim=0) # (num_faces, 3)
        joint_limit_bs = torch.cat(joint_limit_bs, dim=0) # (num_faces, 3)
        joint_types = torch.cat(joint_types, dim=0) # (num_faces, 768)
        
        articulation_features = torch.cat([joint_locations, joint_orientations, joint_limit_bs, joint_types], dim=1).to(torch.float32) # (num_faces, 777)
        target_articulation_existance_labels = torch.from_numpy(np.asarray(articulation_existance_labels)).long() # (num_faces)
        target_articulation_joint_types_labels = torch.from_numpy(np.asarray(joint_type_labels)).long() # (num_faces)

        if self.ce_output:
            target = quantize_coordinates(target, self.num_tokens)
        return features, target, vertices, faces, face_neighborhood, semantic_label_features, target_semantic, articulation_features, target_articulation_existance_labels, target_articulation_joint_types_labels, geometry_features

    
    def get(self, idx):
        # triangle soup with semantic labels
        features, target, _, _, face_neighborhood, semantic_features, semantic_targets = self.get_all_features_for_shape(idx)

        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             semantic_features=semantic_features, semantic_targets=semantic_targets)


class TriangleNodesWithFaces(TriangleNodes):

    def __init__(self, config, split, scale_augment, shift_augment, force_category, face_order_augment, joint_augment):
        super().__init__(config, split, scale_augment, shift_augment, force_category, face_order_augment=face_order_augment, joint_augment=joint_augment)

    # #NOTE: for whole shape structure training: e.g., train structure vqvae or structure transformer
    # def get(self, idx):
    #     # part structure as triangle soup of bbox and semenatic labels
    #     features, target, vertices, faces, face_neighborhood, semantic_features, semantic_targets, articulation_features, articulation_existance_labels, articulation_joint_types_labels, geometry_features = self.get_all_features_for_shape_pc(idx)
    #     return GeometricData(x=torch.from_numpy(features).float(), y=target, mesh_name=self.names[idx],
    #                          edge_index=torch.from_numpy(face_neighborhood.T).long(),
    #                          num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long(),
    #                          semantic_features=semantic_features, semantic_targets=semantic_targets,
    #                          articulation_features=articulation_features, articulation_existance_labels=articulation_existance_labels, articulation_joint_types_labels=articulation_joint_types_labels, geometry_features=geometry_features)
    
    # #NOTE: get a part for structure guided geometry transformer
    def get(self, idx):
        unique_face_labels = self.get_structure_ordering(idx)
        part_id = np.random.choice(unique_face_labels)
        
        features, target, vertices, faces, face_neighborhood, semantic_features, semantic_targets, articulation_features, articulation_existance_labels, articulation_joint_types_labels, geometry_features = self.get_all_features_for_part_pc(idx, part_id=part_id)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, mesh_name=self.names[idx],
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long(),
                             semantic_features=semantic_features, semantic_targets=semantic_targets,
                             articulation_features=articulation_features, articulation_existance_labels=articulation_existance_labels, articulation_joint_types_labels=articulation_joint_types_labels, geometry_features=geometry_features)
        
    
    def get_part(self, idx, part_id):
        
        unique_face_labels = self.get_structure_ordering(idx)
        # Randomly choose a part if part_id is not provided
        if part_id is None:
            part_id = np.random.choice(unique_face_labels)
        
        # Ensure part_id is valid
        if part_id not in unique_face_labels:
            raise ValueError(f"Invalid part_id: {part_id}. Available part IDs are: {unique_face_labels}")
        
        features, target, vertices, faces, face_neighborhood, semantic_features, semantic_targets, articulation_features, articulation_existance_labels, articulation_joint_types_labels, geometry_features = self.get_all_features_for_part_pc(idx, part_id)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, mesh_name=self.names[idx],
                             edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             num_vertices=vertices.shape[0], faces=torch.from_numpy(np.array(faces)).long(),
                             semantic_features=semantic_features, semantic_targets=semantic_targets,
                             articulation_features=articulation_features, articulation_existance_labels=articulation_existance_labels, articulation_joint_types_labels=articulation_joint_types_labels, geometry_features=geometry_features)


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

    def __init__(self, config, split, scale_augment, shift_augment, force_category, joint_augment):
        super().__init__(config, split, scale_augment=scale_augment, shift_augment=shift_augment, force_category=force_category, joint_augment=joint_augment)
        vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.vq_depth = vq_cfg.embed_levels
        self.block_size = config.block_size
        max_inner_face_len = 0
        self.padding = int(config.padding * self.block_size)
        self.sequence_stride = config.sequence_stride
        for i in range(len(self.cached_vertices)):
            self.cached_vertices[i] = np.array(self.cached_vertices[i])
            for j in range(len(self.cached_faces[i])):
                max_inner_face_len = max(
                    max_inner_face_len, len(self.cached_faces[i][j]))
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
            sequence_len = len(
                self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        features, target, _, _, face_neighborhood = self.get_all_features_for_shape(i)
        return GeometricData(x=torch.from_numpy(features).float(), y=target, edge_index=torch.from_numpy(face_neighborhood.T).long(), js=torch.tensor(j).long())

    def plot_sequence_lenght_stats(self):
        sequence_lengths = []
        for i in range(len(self.cached_faces)):
            sequence_len = len(
                self.cached_faces[i]) * self.vq_depth * self.vq_depth_factor + 1 + 1
            sequence_lengths.append(sequence_len)
        import matplotlib.pyplot as plt
        plt.hist(sequence_lengths, bins=32)
        plt.ylim(0, 100)
        plt.show()
        return sequence_lengths


class TriangleNodesWithFacesAndSequenceIndices(TriangleNodesWithSequenceIndices):
    vq_depth_factor = 3

    def __init__(self, config, split, scale_augment, shift_augment, force_category, joint_augment):
        super().__init__(config, split, scale_augment, shift_augment, force_category, joint_augment)

    def get(self, idx):
        i, j, randomness = self.sequence_indices[idx]
        features, target, vertices, faces, face_neighborhood, semantic_features, semantic_targets, articulation_features, articulation_existance_labels, articulation_joint_types_labels, geometry_features = self.get_all_features_for_shape_pc(idx)
        
        return GeometricData(x=torch.from_numpy(features).float(),
                             y=target, mesh_name=self.names[i], edge_index=torch.from_numpy(face_neighborhood.T).long(),
                             js=torch.tensor(j).long(), num_vertices=vertices.shape[0],
                             faces=torch.from_numpy(np.array(faces)).long(),
                             semantic_features=semantic_features, 
                             semantic_targets=semantic_targets,
                             articulation_features=articulation_features, 
                             articulation_existance_labels=articulation_existance_labels, 
                             articulation_joint_types_labels=articulation_joint_types_labels, 
                             geometry_features=geometry_features)
                            

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
    triangles, normals, areas, angles = create_feature_stack_from_triangles(
        triangles)
    return triangles, normals, areas, angles, vertices, faces


def create_feature_stack_from_triangles(triangles):
    t_areas = area(triangles) * 1e3
    t_angles = angle(triangles) / float(np.pi)
    t_normals = unit_vector(normal(triangles))
    return triangles.reshape(-1, 9), t_normals.reshape(-1, 3), t_areas.reshape(-1, 1), t_angles.reshape(-1, 3)


def get_aabb_corners(min_coords, max_coords):
    corners = []
    for x in (min_coords[0], max_coords[0]):
        for y in (min_coords[1], max_coords[1]):
            for z in (min_coords[2], max_coords[2]):
                corners.append((x, y, z))
    return corners


def get_aabb_corners_with_connectivity(aabb_min, aabb_max):
    aabb_verts = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],  # 0: minimum coordinates
        [aabb_max[0], aabb_min[1], aabb_min[2]],  # 1: x_max, y_min, z_min
        [aabb_max[0], aabb_max[1], aabb_min[2]],  # 2: x_max, y_max, z_min
        [aabb_min[0], aabb_max[1], aabb_min[2]],  # 3: x_min, y_max, z_min
        [aabb_min[0], aabb_min[1], aabb_max[2]],  # 4: x_min, y_min, z_max
        [aabb_max[0], aabb_min[1], aabb_max[2]],  # 5: x_max, y_min, z_max
        [aabb_max[0], aabb_max[1], aabb_max[2]],  # 6: maximum coordinates
        [aabb_min[0], aabb_max[1], aabb_max[2]]   # 7: x_min, y_max, z_max
    ])

    # Define the connectivities between corners
    connectivity = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    return aabb_verts, connectivity


def swap_axis(points, num_tokens, clip):
    points = np.clip((points + 0.5), 0, 1) * num_tokens  # type: ignore
    points_quantized_ = points.round().astype(int)

    points_quantized = points_quantized_[:, [2, 0, 1]]
    points_quantized = np.stack([points_quantized[:, 2], points_quantized[:, 1], points_quantized[:, 0]], axis=-1)
    if clip:
        points = points_quantized / num_tokens - 0.5
    else:
        points = points_quantized / num_tokens - 0.5
        points = np.where(points < 0.5, 0, 1)
    # order: Z, Y, X --> X, Y, Z
    points = np.stack([points[:, 2], points[:, 1], points[:, 0]], axis=-1)
    return points

def swap_axis_orientation(points):
    points_quantized_ = points.astype(int) # 0, 1, 0
    points_quantized = points_quantized_[:, [2, 0, 1]] # 0,0,1
    points_quantized = np.stack([points_quantized[:, 2], points_quantized[:, 1], points_quantized[:, 0]], axis=-1) # 1,0,0
    points = np.stack([points_quantized[:, 2], points_quantized[:, 1], points_quantized[:, 0]], axis=-1) 
    return points

def rotate_bbox(vertices, joint_location, joint_orientation, rotation_angle):
    """
    Rotates the bounding box vertices around the given joint.

    :param vertices: numpy array of shape (8, 3) representing the vertices of the bounding box
    :param joint_location: numpy array of shape (3,) representing the location of the joint
    :param joint_orientation: numpy array of shape (3,) representing the axis of rotation as a unit vector
    :param rotation_angle: scalar value representing the angle of rotation around the joint's axis (in radians)
    :return: numpy array of shape (8, 3) representing the rotated vertices
    """
    # Ensure the joint_orientation is a unit vector
    joint_orientation = joint_orientation / np.linalg.norm(joint_orientation)
    
    # Convert axis-angle to rotation matrix
    rotation_matrix = R.from_rotvec(joint_orientation * rotation_angle).as_matrix()

    # Translate vertices to joint location
    translated_vertices = vertices - joint_location

    # Rotate the translated vertices
    rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)

    # Translate back to the original location
    rotated_bbox_vertices = rotated_vertices + joint_location

    return rotated_bbox_vertices
