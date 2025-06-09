from trimesh.exchange.obj import _parse_vertices
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import networkx as nx

from util.misc import normalize_vertices

newface_token = 0
stopface_token = 1
padface_token = 2


def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices


def inner_sort_faces(face):
    min_idx = np.argmin(face)
    face = [face[(i + min_idx) % len(face)] for i, x in enumerate(face)]
    return face


def quantize_vertices_and_faces(vertices_, faces_, max_vertices, max_faces, num_tokens):
    def face_to_cycles(face):
        """Find cycles in face."""
        g = nx.Graph()
        for v in range(len(face) - 1):
            g.add_edge(face[v], face[v + 1])
        g.add_edge(face[-1], face[0])
        return list(nx.cycle_basis(g))

    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  # type: ignore
    vertices_quantized_ = vertices.round().astype(int)

    if vertices_quantized_.shape[0] > max_vertices:
        raise ValueError("Vertices exceed max vertices:", vertices_quantized_.shape[0], max_vertices)
    if len([x for fl in faces_ for x in fl]) > max_faces:
        raise ValueError("Faces exceed max faces:", len([x for fl in faces_ for x in fl]), max_faces)

    vertices_quantized_ = vertices_quantized_[:, [2, 0, 1]]
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)

    sort_inds = np.lexsort(vertices_quantized.T)

    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_]

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices_quantized.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices_quantized = vertices_quantized[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]

    face_sequence = []
    facepos_inner = []
    facepos_outer = []

    for fi, face in enumerate(faces):
        face_sequence.append(newface_token)
        facepos_inner.append(newface_token)
        facepos_outer.append(newface_token)
        for vi, v in enumerate(face):
            face_sequence.append(v + 3)
            facepos_inner.append(vi + 3)
            facepos_outer.append(fi + 3)
    face_sequence.append(stopface_token)
    facepos_inner.append(stopface_token)
    facepos_outer.append(stopface_token)

    return vertices_quantized, faces, face_sequence, facepos_inner, facepos_outer


def process_mesh(mesh_path, num_tokens, max_vertices, max_faces):
    text = Path(mesh_path).read_text()
    vertices = normalize_vertices(_parse_vertices(text)[0])
    faces_ = read_faces(text)
    return quantize_vertices_and_faces(vertices, faces_, max_vertices, max_faces, num_tokens)


def create_face_dataset_file(train, val, output_file, num_tokens, max_vertices, max_faces):
    valid_names_train = []
    cached_vertices_train = []
    cached_faces_train = []
    valid_names_val = []
    cached_vertices_val = []
    cached_faces_val = []
    # names = [x for x in names if x.stem.endswith("dec05")]
    for p in tqdm(train, desc='data preload'):
        try:
            text = p.read_text()
            mesh_vertices = _parse_vertices(text)[0]
            mesh_faces = read_faces(text)
            quantize_vertices_and_faces(mesh_vertices, mesh_faces, num_tokens=num_tokens, max_vertices=max_vertices, max_faces=max_faces)  # throws error if processing fails
            cached_vertices_train.append(mesh_vertices)
            cached_faces_train.append(mesh_faces)
            valid_names_train.append(p.stem)
        except Exception as err:
            print('Exception occured while processing', p, 'skipping...', err)
    for p in tqdm(val, desc='data preload'):
        try:
            text = p.read_text()
            mesh_vertices = _parse_vertices(text)[0]
            mesh_faces = read_faces(text)
            quantize_vertices_and_faces(mesh_vertices, mesh_faces, num_tokens=num_tokens, max_vertices=max_vertices, max_faces=max_faces)  # throws error if processing fails
            cached_vertices_val.append(mesh_vertices)
            cached_faces_val.append(mesh_faces)
            valid_names_val.append(p.stem)
        except Exception as err:
            print('Exception occured while processing', p, 'skipping...', err)

    print('Train:', len(cached_vertices_train))
    print('Val:', len(cached_vertices_val))
    pickle.dump({
        'name_train': valid_names_train,
        'vertices_train': cached_vertices_train,
        'faces_train': cached_faces_train,
        'name_val': valid_names_val,
        'vertices_val': cached_vertices_val,
        'faces_val': cached_faces_val,
    }, open(f'data/shapenet/{output_file}.pkl', 'wb'))



if __name__ == "__main__":
    pass
