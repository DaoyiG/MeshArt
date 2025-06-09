

import numpy as np
import torch
from collections import OrderedDict
from .meshlab import meshlab_proc
import trimesh
import itertools
from scipy.spatial.distance import cdist

def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith("f ")]
    all_faces = [[int(y.split("/")[0]) - 1 for y in x.strip().split(" ")[1:]] for x in all_face_lines]
    return all_faces

def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith("v ")]
    all_vertices = np.array([[float(y) for y in x.strip().split(" ")[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, "vertices should have 3 coordinates"
    return all_vertices

def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence

def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + '.', '')] = state_dict[k]
    return new_state_dict

def sem_accuracy(y_pred, y_true, ignore_label=None, device=None):
    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (torch.tensor(y_pred, device=device).reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer

def accuracy(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=-1)

    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer

def rmse(y_pred, y_true, num_tokens, ignore_labels=(0, 1, 2)):
    mask = torch.logical_and(y_true != ignore_labels[0], y_pred != ignore_labels[0])
    for i in range(1, len(ignore_labels)):
        mask = torch.logical_and(mask, y_true != ignore_labels[i])
        mask = torch.logical_and(mask, y_pred != ignore_labels[i])
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    vertices_pred = (y_pred - 3) / num_tokens - 0.5
    vertices_true = (y_true - 3) / num_tokens - 0.5
    return torch.sqrt(torch.mean((vertices_pred - vertices_true)**2))


def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices

def scale_vertices_returnscale(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices, x, y, z

def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices

def shift_vertices_returnshifts(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices, x, y, z

def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices

def normalize_vertices_returnbounds(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices, bounds

def top_p_sampling(logits, p):
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def query_junction_from_pool(part_bbox, junction_pool, distance_threshold=0.1):
    
    query_triangles = junction_pool['triangles'] # list of np arrays
    query_tokens = junction_pool['tokens'] # list of ints, 6 times the number of triangles
    query_points = []
    query_triangle_centroids = []
    for f_idx, f in enumerate(query_triangles):
        query_points.extend(f.reshape(-1, 3))
        query_triangle_centroids.append(f.mean(axis=0))
    query_points = np.array(query_points)
    query_triangle_centroids = np.array(query_triangle_centroids)
    _, dists, _ = trimesh.proximity.closest_point(part_bbox, query_points)
    dists_per_face = dists.reshape(-1, 3)
    min_dists = np.min(dists_per_face, axis=1)
    junction_flags = min_dists < distance_threshold
    junction_face_ids = np.where(junction_flags)[0]
    
    sorted_indices = np.argsort(query_triangle_centroids[junction_flags, 2])
    junction_face_ids = junction_face_ids[sorted_indices]
    
    junction_tokens = []
    for jf_idx in junction_face_ids:
        junction_tokens.extend(query_tokens[jf_idx*6:(jf_idx+1)*6])

    selected_junction_triangles = [query_triangles[i] for i in junction_face_ids]
    return selected_junction_triangles, junction_tokens

def get_functional_part_ordering(mesh_category, vertices, faces, face_labels):
    
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
        uid_heights[uid] = part_bbox_verts[:, 2].min()
    
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

def is_box_like(vertices, faces, relative_tolerance=0.1):
    vertices = np.array(vertices).reshape(-1, 3)
    
    # Calculate the bounding box
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    
    # Calculate the box diagonal length
    box_diagonal = np.linalg.norm(max_bounds - min_bounds)
    
    # Set the tolerance relative to the box size
    tolerance = relative_tolerance * box_diagonal
    
    # Check if each vertex is close to a corner of the bounding box
    corners = np.array(list(itertools.product(*zip(min_bounds, max_bounds))))
    distances = cdist(vertices, corners)
    if not np.all(np.min(distances, axis=1) < tolerance):
        return False
    
    return True

def functional_structure_mapping(sequence_semtok_dict, category, part_sequence_dict_idx, gen_face_arti_type, coords_arti_loc, coords_arti_ori):
    functional_part_mapping = {-1:[]}
    functional_joint_mapping = {}
    for uid in sequence_semtok_dict: # per-semantic labels
        if category == '02933112':
            num_drawer, num_door = 0, 0
            if uid < 3:
                functional_part_mapping[-1].extend(sequence_semtok_dict[uid])
                functional_joint_mapping[-1] = {
                        'exist': 0,
                        'type': 0,
                        'loc': np.zeros(3),
                        'ori': np.zeros(3)
                    }
            else:
                if uid == 3:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_door
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72] # no start token, end token
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_door += 1
                    
                elif uid == 4:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_drawer
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72]
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_drawer += 1
                else:
                    raise ValueError(f"Unknown part id: {uid}")
        elif category == '04379243':
            num_door, num_drawer, num_wheel = 0, 0, 0
            if uid < 2 or uid == 4:
                functional_part_mapping[-1].extend(sequence_semtok_dict[uid])
                functional_joint_mapping[-1] = {
                        'exist': 0,
                        'type': 0,
                        'loc': np.zeros(3),
                        'ori': np.zeros(3)
                    }
            else:
                if uid == 2:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_drawer
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72]
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_drawer += 1
                elif uid == 3:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_door
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72]
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_door += 1
                elif uid == 5:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_wheel
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72]
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_wheel += 1
                else:
                    raise ValueError(f"Unknown part id: {uid}")
        elif category == '03001627':
            num_wheel = 0
            if uid < 5 and uid != 0:
                functional_part_mapping[-1].extend(sequence_semtok_dict[uid])
                functional_joint_mapping[-1] = {
                        'exist': 0,
                        'type': 0,
                        'loc': np.zeros(3),
                        'ori': np.zeros(3)
                    }
            else:
                if uid == 5:
                    num_instances = len(sequence_semtok_dict[uid]) // 72
                    for i in range(num_instances):
                        class_inst = uid * 100 + num_wheel
                        functional_part_mapping[class_inst] = sequence_semtok_dict[uid][i*72:(i+1)*72]
                        part_joint_type = []
                        part_joint_loc = []
                        part_joint_ori = []
                        idx_range = part_sequence_dict_idx[uid][i*12:(i+1)*12]
                        for idx in idx_range:
                            part_joint_type.append(gen_face_arti_type[idx])
                            part_joint_loc.append(coords_arti_loc[idx])
                            part_joint_ori.append(coords_arti_ori[idx])
                        functional_joint_mapping[class_inst] = {
                            'exist': 1,
                            'type': np.array(part_joint_type).mean(),
                            'loc': np.array(part_joint_loc).mean(axis=0),
                            'ori': np.array(part_joint_ori).mean(axis=0)
                        }
                        num_wheel += 1
                elif uid == 0:
                    functional_part_mapping[uid] = sequence_semtok_dict[uid]
                    part_joint_type = []
                    part_joint_loc = []
                    part_joint_ori = []
                    for idx in part_sequence_dict_idx[uid]:
                        part_joint_type.append(gen_face_arti_type[idx])
                        part_joint_loc.append(coords_arti_loc[idx])
                        part_joint_ori.append(coords_arti_ori[idx])
                    functional_joint_mapping[uid] = {
                        'exist': 1,
                        'type': np.array(part_joint_type).mean(),
                        'loc': np.array(part_joint_loc).mean(axis=0),
                        'ori': np.array(part_joint_ori).mean(axis=0)
                    }
                else:
                    raise ValueError(f"Unknown part id: {uid}")
        else:
            raise ValueError(f"Unknown category: {category}")
    return functional_part_mapping, functional_joint_mapping

# naive generation test wrt generated geometry with its part bbox
def naive_gen_test(part_mesh, part_bbox_mesh, voxel_size):
    valid = True
    current_voxels = part_mesh.voxelized(voxel_size)
    current_points = current_voxels.points
    current_total_voxels = len(current_points)
    bbox_voxels = part_bbox_mesh.voxelized(voxel_size)
    bbox_total_voxels = len(bbox_voxels.points)
    bbox_occupancy_ratio = current_total_voxels / bbox_total_voxels
    
    if bbox_occupancy_ratio <= 0.15 or bbox_occupancy_ratio >= 1.5:
        valid = False
        print(f'generated part is too small or too big compared to the bbox with {bbox_occupancy_ratio}')
    return valid

