import os
import random
import sys
import omegaconf
import torch
from pathlib import Path
os.environ['PYOPENGL_PLATFORM'] = 'egl'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import trimesh
from dataset.quantized_soup_structure import QuantizedSoupTripletsCreator
from dataset.triangles_structure import TriangleNodesWithFacesAndSequenceIndices
from trainer import get_rvqvae_v4_decoder
from trainer.train_structure_transformer import get_qsoup_model_config
from util.misc import get_parameters_from_state_dict
from util.visualization import plot_vertices_and_faces_withfacelabels, plot_vertices_and_faces_withfacelabels_wjoint, plot_vertices_and_faces_withfacelabels_wjointonpart
from model.transformer_structure import QuantSoupTransformer as QuantSoupTransformerNoEmbed
from pytorch_lightning import seed_everything
import numpy as np
import json
import pickle
from util.misc import is_box_like, functional_structure_mapping



@torch.no_grad()
def main(config, mode, start_idx):
    seed_everything(42)

    device = torch.device('cuda:0')
    vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")

    is_val = False

    train_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', False, False, config.ft_category, False)
    val_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'val', False, False, config.ft_category, False)
    dataset = val_dataset if is_val else train_dataset

    num_kickoff_triangles = 1
    
    print(f'Saving to: {config.root_dir}/{config.experiment}/inference')
    
    output_dir_image_rigid = Path(f'{config.root_dir}/{config.experiment}/inference/inf_image_rigid_{mode}')
    output_dir_image_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_image_articulated = Path(f'{config.root_dir}/{config.experiment}/inference/inf_image_articulated_{mode}')
    output_dir_image_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_part_rigid = Path(f'{config.root_dir}/{config.experiment}/inference/inf_part_rigid_{mode}')
    output_dir_part_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_part_articulated = Path(f'{config.root_dir}/{config.experiment}/inference/inf_part_articulated_{mode}')
    output_dir_part_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_mesh_rigid = Path(f'{config.root_dir}/{config.experiment}/inference/inf_mesh_rigid_{mode}')
    output_dir_mesh_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_mesh_articulated = Path(f'{config.root_dir}/{config.experiment}/inference/inf_mesh_articulated_{mode}')
    output_dir_mesh_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_face_labels_rigid = Path( f'{config.root_dir}/{config.experiment}/inference/inf_face_labels_rigid_{mode}')
    output_dir_face_labels_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_face_labels_articulated = Path( f'{config.root_dir}/{config.experiment}/inference/inf_face_labels_articulated_{mode}')
    output_dir_face_labels_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_joints_rigid = Path( f'{config.root_dir}/{config.experiment}/inference/inf_joints_rigid_{mode}')
    output_dir_joints_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_joints_articulated = Path( f'{config.root_dir}/{config.experiment}/inference/inf_joints_articulated_{mode}')
    output_dir_joints_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_pkls_rigid = Path( f'{config.root_dir}/{config.experiment}/inference/inf_pkls_rigid_{mode}')
    output_dir_pkls_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_pkls_articulated = Path( f'{config.root_dir}/{config.experiment}/inference/inf_pkls_articulated_{mode}')
    output_dir_pkls_articulated.mkdir(exist_ok=True, parents=True)
    
    out_dir_funcional_mapping = Path(f'{config.root_dir}/{config.experiment}/inference/inf_funcional_mapping')
    out_dir_funcional_mapping.mkdir(exist_ok=True, parents=True)

    model_cfg = get_qsoup_model_config(config, vq_cfg.embed_levels)
    model = QuantSoupTransformerNoEmbed(model_cfg, vq_cfg)
    state_dict = torch.load(config.resume, map_location="cpu")["state_dict"]
    sequencer = QuantizedSoupTripletsCreator(config, vq_cfg)
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device)
    model = model.eval()
    sequencer = sequencer.to(device)
    sequencer = sequencer.eval()
    decoder = get_rvqvae_v4_decoder(vq_cfg, config.vq_resume, device)

    kickoff_tokens = 1 + 6 * num_kickoff_triangles
    articulated_count = 0
    
    while articulated_count < config.num_val_samples:
        rand_idx = random.randint(0, len(dataset) - 1)
        data = dataset.get(rand_idx)
        
        print(f"data {rand_idx}, name {dataset.names[rand_idx]}")

        soup_sequence, face_in_idx, face_out_idx, target = sequencer.get_completion_sequence(
            data.x.to(device),
            data.edge_index.to(device),
            data.faces.to(device),
            data.num_vertices,
            kickoff_tokens,
            data.semantic_features.to(device),
            data.geometry_features.to(device),
            data.articulation_features.to(device)
        )
        y = None

        if mode == "topp":
            y = model.generate(
                soup_sequence, face_in_idx, face_out_idx, sequencer, config.max_val_tokens,
                temperature=config.temperature, top_k=config.top_k_tokens, top_p=config.top_p, use_kv_cache=False,
            )
        elif mode == "beam":
            y = model.generate_with_beamsearch(
                soup_sequence, face_in_idx, face_out_idx, sequencer, config.max_val_tokens, use_kv_cache=True, beam_width=6
            )
        if y is None:
            continue
        
        gen_vertices, gen_faces, gen_face_labels, decoded_x_conv_geo, gen_face_arti_exist, gen_face_arti_type, coords_arti_loc, coords_arti_ori, sequence_semtok_dict, part_sequence_dict_idx = sequencer.decode(y[0], decoder, return_semantics=True, return_geo_feat=True, return_articulation_info=True, semantic_retrieval_array=torch.tensor(dataset.semantic_retrieval_array), semantic_feature_decipher=dataset.semantic_feature_decipher, return_token_dict=True)

        # filtering strategies:
        # 1. generated part structure collapse: generally the volumn of the generated part and its bbox should be similar, but if the generated triangles does not form a box, then its bbox volumn should be larger than the generated part, filter this out
        # 2. if a part has joint predicted but the joint locations differs a lot on the 12 faces of that part, then filter this out
        # Filtering strategies
        # 1. Check for box-like structure
        is_valid = True
        for i in range(0, len(gen_faces), 12):
            part_vertices = gen_vertices[gen_faces[i:i+12]]
            part_faces = gen_faces[i:i+12]
            if not is_box_like(part_vertices, part_faces):
                is_valid = False
                break

        # 2. Check for inconsistent joint locations
        if is_valid:
            for i in range(0, len(gen_face_arti_type), 12):
                if any(gen_face_arti_type[i:i+12]):
                    joint_locs = coords_arti_loc[i:i+12]
                    mean_loc = np.mean(joint_locs, axis=0)
                    distances = np.linalg.norm(joint_locs - mean_loc, axis=1)
                    if np.max(distances) > 0.1:  # Threshold for inconsistency
                        is_valid = False
                        break

        if not is_valid:
            print(f"Filtered out sample {rand_idx:06d}")
            continue
        
        arti_exist = np.array(gen_face_arti_exist)
        if sum(arti_exist) >= 10:
            articulated = True
            output_dir_image = output_dir_image_articulated
            output_dir_mesh = output_dir_mesh_articulated
            output_dir_face_labels = output_dir_face_labels_articulated
            output_dir_joints = output_dir_joints_articulated
            output_dir_pkls = output_dir_pkls_articulated
            output_dir_part = output_dir_part_articulated
        else:
            output_dir_image = output_dir_image_rigid
            output_dir_mesh = output_dir_mesh_rigid
            output_dir_face_labels = output_dir_face_labels_rigid
            output_dir_joints = output_dir_joints_rigid
            output_dir_pkls = output_dir_pkls_rigid
            output_dir_part = output_dir_part_rigid
            articulated = False
        
        if not articulated: # skip non-articulated object
            continue
        
        functional_part_mapping = {-1:[]}
        functional_joint_mapping = {}
        if articulated:
            functional_part_mapping, functional_joint_mapping = functional_structure_mapping(sequence_semtok_dict, config.ft_category, part_sequence_dict_idx, gen_face_arti_type, coords_arti_loc, coords_arti_ori)
        else:
            functional_part_mapping[-1] = y[0].detach().cpu().numpy().tolist()[1:-1]
            functional_joint_mapping[-1] = {
                'exist': 0,
                'type': 0,
                'loc': np.zeros(3),
                'ori': np.zeros(3)
            }
        functional_part_mapping['shape'] = y[0].detach().cpu().numpy().tolist()[1:-1]
        with open(out_dir_funcional_mapping / f"functional_part_mapping_{articulated_count}.pt", "wb") as f:
            torch.save(functional_part_mapping, f)
        with open(out_dir_funcional_mapping / f"functional_joint_mapping_{articulated_count}.pt", "wb") as f:
            torch.save(functional_joint_mapping, f)
            
        functional_triangles = {}
        whole_verts = []
        whole_faces = []
        whole_face_labels = []
        for uid in functional_part_mapping:
            if uid != 'shape':
                # Get structure sequence for this functional part
                part_sequence = torch.tensor(functional_part_mapping[uid], device=device)
                # Decode structure sequence to get bounding box mesh
                bbox_vertices, bbox_faces, bbox_face_labels = sequencer.decode(part_sequence, decoder, return_semantics=True, semantic_retrieval_array=torch.tensor(dataset.semantic_retrieval_array), semantic_feature_decipher=dataset.semantic_feature_decipher)
                # Store vertices and faces with functional uid labels
                functional_triangles[uid] = {
                    'vertices': bbox_vertices,
                    'faces': bbox_faces,
                    'face_labels': [uid] * len(bbox_faces) # Label all faces with the functional uid
                }
                whole_faces.extend( [(np.asarray(f)+len(whole_verts)).tolist()  for f in bbox_faces])
                whole_verts.extend(bbox_vertices)
                whole_face_labels.extend([uid] * len(bbox_faces))
        whole_verts = np.asarray(whole_verts)
        functional_triangles['shape'] = {
            'vertices': whole_verts,
            'faces': whole_faces,
            'face_labels': whole_face_labels
        }
        color_dict = {v: (random.random(), random.random(), random.random()) for v in np.unique(whole_face_labels)}
        plot_vertices_and_faces_withfacelabels(whole_verts, whole_faces, whole_face_labels, output_dir_part / f"{start_idx + articulated_count:06d}_shape_structure_functionaluid.jpg", color=color_dict)

        with open(out_dir_funcional_mapping / f"functional_triangles_{articulated_count}.pt", "wb") as f:
            torch.save(functional_triangles, f)
        
        for uid in functional_part_mapping:
            part_sequence = torch.tensor(functional_part_mapping[uid], device=device)
            gen_part_vertices, gen_part_faces = sequencer.decode(part_sequence, decoder, return_semantics=False, return_geo_feat=False, return_articulation_info=False, semantic_retrieval_array=torch.tensor(dataset.semantic_retrieval_array), semantic_feature_decipher=dataset.semantic_feature_decipher, return_token_dict=False)
            pseudo_face_labels = np.arange(len(gen_part_faces))
            num_faces = len(gen_part_faces)
            color_dict = {}
            for i in range(num_faces):
                r = int(255 - (i * (255-102) / num_faces))  # Red goes from 255 to 102
                g = int(230 - (i * 230 / num_faces))  # Green goes from 230 to 0  
                b = int(230 - (i * 230 / num_faces))  # Blue goes from 230 to 0
                color_dict[i] = f'#{r:02x}{g:02x}{b:02x}'
            plot_vertices_and_faces_withfacelabels(gen_part_vertices, gen_part_faces, pseudo_face_labels, output_dir_part / f"{start_idx + articulated_count:06d}_part_{uid}.jpg", color=color_dict)
            
        with open(output_dir_pkls / f"{start_idx + articulated_count:06d}.pkl", "wb") as f:
            pickle.dump(y.detach().cpu().numpy(), f)

        plot_vertices_and_faces_withfacelabels(gen_vertices, gen_faces, gen_face_labels, output_dir_image / f"{start_idx + articulated_count:06d}.jpg")

        # save face labels
        with open(output_dir_face_labels / f"{start_idx + articulated_count:06d}.txt", "w") as f:
            for label in gen_face_labels:
                f.write(f"{label}\n")
                
        num_face_labels_gen = np.unique(gen_face_labels)
        colors_gen = {v: (random.random(), random.random(), random.random()) for v in (num_face_labels_gen)}
        
        plot_vertices_and_faces_withfacelabels_wjoint(gen_vertices, gen_faces, gen_face_labels, coords_arti_loc, coords_arti_ori, gen_face_arti_type, output_dir_image / f"{start_idx + articulated_count:06d}_gen_joint.jpg", color=colors_gen)

        articulation_infos = {'joint_locs': coords_arti_loc.tolist(), 'joint_oris': coords_arti_ori.tolist(), 'joint_types': gen_face_arti_type}

        # for prismatic joint (joint type 2), change the joint location to the centroid of the bbox of the joint, visualize the each joint on the bbox
        # the order of the joint is the same as the order of the faces; calculate the centroid of every 12 faces
        centroid_coords = []
        for i in range(0, len(gen_faces), 12):
            centroid_coords.append(np.mean(gen_vertices[gen_faces[i:i+12]], axis=0))
        new_joint_locs = []
        for i in range(0, len(gen_face_arti_type)):
            bbox_num = i // 12
            if gen_face_arti_type[i] == 2:
                new_joint_loc = centroid_coords[bbox_num]
                new_joint_locs.append(new_joint_loc)
            else:
                new_joint_locs.append(coords_arti_loc[i])
                
        bbox_inst_face_labels = np.zeros(len(gen_faces))
        joint_types = np.zeros(len(gen_faces))
        # every 12 faces assign a label
        for i in range(len(gen_faces)):
            bbox_inst_face_labels[i] = i // 12
            joint_types[i] = i // 12
        bbox_inst_face_colors = {v: (random.random(), random.random(), random.random()) for v in np.unique(bbox_inst_face_labels)}
        plot_vertices_and_faces_withfacelabels_wjointonpart(gen_vertices, gen_faces, bbox_inst_face_labels, new_joint_locs, coords_arti_ori, joint_types, output_dir_image / f"{start_idx + articulated_count:06d}_gen_jointonpart.jpg", color=bbox_inst_face_colors)
        if sum(np.array(gen_face_arti_type)) > 0:
            articulation_info_file = output_dir_joints / f"{start_idx + articulated_count:06d}.json"
            with open(articulation_info_file, 'w') as f:
                json.dump(articulation_infos, f, indent=2)
        
        trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False, validata=False, maintain_order=True).export(output_dir_mesh / f"{start_idx + articulated_count:06d}.obj")
        articulated_count += 1

        
if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load(Path(sys.argv[1]).parents[1] / "config.yaml")
    cfg.resume = sys.argv[1]
    cfg.padding = 0.0
    cfg.num_val_samples = int(sys.argv[4])
    cfg.sequence_stride = cfg.block_size
    cfg.top_p = 0.95
    cfg.temperature = 1.0
    cfg.train_structure_transformer = True
    cfg.low_augment = True
    cfg.ft_category = sys.argv[5]
    
    # NOTE: need to set correct path to the data and checkpoints
    main(cfg, sys.argv[2], int(sys.argv[3]))
