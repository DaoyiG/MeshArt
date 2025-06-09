import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import warnings
warnings.filterwarnings("ignore")



def plot_vertices(vertices, output_path):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='g', s=10)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_vertices_and_faces(vertices, faces, output_path, color='b'):
    ngons = [[vertices[v, :].tolist() for v in f] for f in faces]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
    polygon_collection = Poly3DCollection(ngons)
    polygon_collection.set_alpha(0.3)
    polygon_collection.set_color(color)
    ax.add_collection(polygon_collection)
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")



def plot_vertices_and_faces_withfacelabels(vertices, faces, face_labels, output_path, color={0: 'black', 1: 'b', 2: 'g', 3: 'r', 4:'chocolate', 5:'cyan', -1: 'gray', 16390: 'seashell', 16391: 'r',16392: 'r',16393: 'r', 16394: 'r', 16395: 'r', 16396: 'r', 16397: 'r', 16398: 'r', 16399: 'r'}):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    unique_labels = sorted(np.unique(face_labels))
    for label in unique_labels:
        # get the ngons
        ngons = [[vertices[v, :].tolist() for v in f] for i, f in enumerate(faces) if face_labels[i] == label]
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=1)
        if label == -1:
            polygon_collection = Poly3DCollection(ngons)
            polygon_collection.set_alpha(0.2)
        else:
            polygon_collection = Poly3DCollection(ngons)
            polygon_collection.set_alpha(0.15)
        polygon_collection.set_color(color[int(label)])
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_vertices_and_faces_withfacelabels_wjointonpart(vertices, faces, face_labels, joint_locations, joint_orientations, joint_types, output_path, color={0: 'gray', 1: 'b', 2: 'g', 3: 'r', 4:'chocolate', 5:'cyan', -1: 'seashell', 16390: 'seashell', 16391: 'r',16392: 'r',16393: 'r'}):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
        
    unique_labels = sorted(np.unique(face_labels)) # class-level
    for label in (unique_labels):
        # get the ngons
        ngons = [[vertices[v, :].tolist() for v in f] for i, f in enumerate(faces) if face_labels[i] == label]
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
        polygon_collection = Poly3DCollection(ngons)
        polygon_collection.set_alpha(0.15)
        polygon_collection.set_color(color[int(label)])
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        
        joints_locs = [joint_locations[j] for j, joint_uid in enumerate(joint_types) if joint_uid == label]
        joint_oris = [joint_orientations[j] for j, joint_uid in enumerate(joint_types) if joint_uid == label]
        for i in range(len(joints_locs)):
            joint_loc = joints_locs[i].reshape(-1)
            joint_ori = joint_oris[i].reshape(-1)
            ax.quiver(joint_loc[0], joint_loc[1], joint_loc[2], joint_ori[0], joint_ori[1], joint_ori[2], length=0.4, normalize=True, color=color[int(label)])
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")

def plot_vertices_and_faces_withfacelabels_wjoint(vertices, faces, face_labels, joint_locations, joint_orientations, joint_types, output_path, color={0: 'gray', 1: 'b', 2: 'g', 3: 'r', 4:'chocolate', 5:'cyan', -1: 'seashell', 16390: 'seashell', 16391: 'r',16392: 'r',16393: 'r'}):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    
    for i in range(len(joint_locations)):
        joint_loc = joint_locations[i].reshape(-1)
        joint_ori = joint_orientations[i].reshape(-1)
        joint_type = joint_types[i]
        if joint_type == 'prismatic' or joint_type == 2:
            ax.quiver(joint_loc[0], joint_loc[1], joint_loc[2], joint_ori[0], joint_ori[1], joint_ori[2], length=0.3, normalize=True, color='cyan')
        elif joint_type == 'revolute' or joint_type == 1:
            ax.quiver(joint_loc[0], joint_loc[1], joint_loc[2], joint_ori[0], joint_ori[1], joint_ori[2], length=0.3, normalize=True, color='orange')
        elif joint_type == 'fixed' or joint_type == 0:
            pass
        else:
            raise ValueError("Unknown joint type")
        
    unique_labels = sorted(np.unique(face_labels))
    for j, label in enumerate(unique_labels):
        # get the ngons
        ngons = [[vertices[v, :].tolist() for v in f] for i, f in enumerate(faces) if face_labels[i] == label]
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='black', s=10)
        polygon_collection = Poly3DCollection(ngons)
        polygon_collection.set_alpha(0.2)
        polygon_collection.set_color(color[int(label)])
        
        ax.add_collection(polygon_collection)
        ax.set_zlim(-0.35, 0.35)
        ax.view_init(25, -120, 0)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")


def plot_combined_vertices_and_faces(vertices_1, faces_1, vertices_2, faces_2, output_path, color_1='b', color_2='r'):
    ngons = [[vertices_1[v, :].tolist() for v in f] for f in faces_1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.xlim(-0.45, 0.45)
    plt.ylim(-0.45, 0.45)
    # Don't mess with the limits!
    plt.autoscale(False)
    ax.set_axis_off()
    ax.scatter(vertices_1[:, 0], vertices_1[:, 1], vertices_1[:, 2], c='black', s=10)
    polygon_collection_1 = Poly3DCollection(ngons)
    polygon_collection_1.set_alpha(0.3)
    polygon_collection_1.set_color(color_1)
    ax.add_collection(polygon_collection_1)
    
    ngons_2 = [[vertices_2[v, :].tolist() for v in f] for f in faces_2]
    ax.scatter(vertices_2[:, 0], vertices_2[:, 1], vertices_2[:, 2], c='gray', s=10)
    polygon_collection_2 = Poly3DCollection(ngons_2)
    polygon_collection_2.set_alpha(0.3)
    polygon_collection_2.set_color(color_2)
    ax.add_collection(polygon_collection_2)
    
    ax.set_zlim(-0.35, 0.35)
    ax.view_init(25, -120, 0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")

def ngon_to_obj(vertices, faces):
    obj = ""
    for i in range(len(vertices)):
        obj += f"v {vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n"
    for i in range(len(faces)):
        fline = "f"
        for j in range(len(faces[i])):
            fline += f" {faces[i][j] + 1} "
        fline += "\n"
        obj += fline
    return obj

def triangle_sequence_to_mesh(triangles):
    vertices = triangles.reshape(-1, 3)
    faces = np.array(list(range(vertices.shape[0]))).reshape(-1, 3)
    return vertices, faces

def triangle_mesh_to_quads(m):
    quads = m.face_adjacency[m.face_adjacency_angles < 1e-3]
    quads = m.faces[m.face_adjacency[m.face_adjacency_angles < 1e-3]]
    return quads

def triangle_sequence_to_mesh_wlabels(triangles):
    vertices_all = triangles.reshape(-1, 3)
    faces_all = np.array(list(range(vertices_all.shape[0]))).reshape(-1, 3)
    face_labels_all = np.zeros(len(faces_all))
    # every 12 faces assign a label
    for i in range(len(faces_all)):
        face_labels_all[i] = i // 12
    return vertices_all, faces_all, face_labels_all

def bounds_sequence_to_bbox(bounds):
    
    N = bounds.shape[0]
    
    vertices_all = []
    faces_all = []
    faces_labels_all = []
    for i in range(N):
        bound = bounds[i]
        bound = bound.reshape(2, 3)
        vertices, faces = get_aabb_corners_with_connectivity(bound[0], bound[1])
        
        bbox_faces_ = [(np.asarray(f)+len(vertices_all)).tolist() for f in faces]

        vertices_all.extend(vertices)
        faces_all.extend(bbox_faces_)
        faces_labels_all.extend([i] * len(faces))
    vertices_all = np.array(vertices_all)
    return vertices_all, faces_all, faces_labels_all

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