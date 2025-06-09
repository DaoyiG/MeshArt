import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import omegaconf
import torch
from pathlib import Path
import trimesh
from dataset.quantized_soup_wjunc import QuantizedSoupTripletsCreator
from dataset.quantized_soup_structure import QuantizedSoupTripletsCreator as QuantizedSoupTripletsCreatorStructure 

from dataset.triangles_geometry import TriangleNodesWithFacesAndSequenceIndices
from trainer import get_rvqvae_v0junc_decoder, get_rvqvae_v4_decoder
from trainer.train_geometry_transformer import get_qsoup_model_config
from util.misc import get_parameters_from_state_dict
from util.visualization import plot_vertices_and_faces_withfacelabels
from model.transformer_geometry import QuantSoupTransformer as QuantSoupTransformerNoEmbed
import numpy as np
from util.meshlab import meshlab_proc
from util.misc import read_faces, read_vertices, query_junction_from_pool, get_functional_part_ordering, naive_gen_test
from pytorch_lightning import seed_everything
from tqdm import tqdm
# from model.pointnet import get_pointnet_classifier

@torch.no_grad()
def main(config, mode, start_idx=0):
    seed_everything(42)
    device = torch.device("cuda:0")
    vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
    vq_cfg_structure = omegaconf.OmegaConf.load(Path(config.vq_resume_structure).parents[1] / "config.yaml")

    dataset = TriangleNodesWithFacesAndSequenceIndices(config, "train", False, False, config.ft_category, joint_augment=config.joint_augment, junction_augment=config.junction_augment)

    model_cfg = get_qsoup_model_config(config, vq_cfg.embed_levels)
    model = QuantSoupTransformerNoEmbed(model_cfg, vq_cfg)
    state_dict = torch.load(config.resume, map_location="cpu")["state_dict"]
    sequencer = QuantizedSoupTripletsCreator(config, vq_cfg)
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device)
    model = model.eval()
    sequencer = sequencer.to(device)
    sequencer = sequencer.eval()
    decoder = get_rvqvae_v0junc_decoder(vq_cfg, config.vq_resume, device)
    
    # pnet = get_pointnet_classifier(config.pnet_ckpt).to(device)

    sequencer_structure = QuantizedSoupTripletsCreatorStructure(config, vq_cfg_structure)
    sequencer_structure = sequencer_structure.to(device)
    sequencer_structure = sequencer_structure.eval()

    decoder_structure = get_rvqvae_v4_decoder(vq_cfg_structure, config.vq_resume_structure, device)
    
    functional_part_mapping_dir = Path(config.structure_functional_part_mapping_dir)
    articulated_structure_mesh_dir = Path(config.articulated_structure_mesh_dir)
    
    print(f'Saving to: {config.root_dir}/{config.experiment}/inference')
    output_dir_image_rigid = Path(f"{config.root_dir}/{config.experiment}/inference/inf_image_rigid_{mode}")
    output_dir_image_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_image_articulated = Path(f"{config.root_dir}/{config.experiment}/inference/inf_image_articulated_{mode}")
    output_dir_image_articulated.mkdir(exist_ok=True, parents=True)
    output_dir_mesh_rigid = Path(f"{config.root_dir}/{config.experiment}/inference/inf_mesh_rigid_{mode}")
    output_dir_mesh_rigid.mkdir(exist_ok=True, parents=True)
    output_dir_mesh_articulated = Path(f"{config.root_dir}/{config.experiment}/inference/inf_mesh_articulated_{mode}")
    output_dir_mesh_articulated.mkdir(exist_ok=True, parents=True)

    gen_pool = os.listdir(articulated_structure_mesh_dir)
    gen_list = list(set([int(x.split('.')[0].split('_')[-1]) for x in gen_pool]))
    
    
    for k in tqdm(gen_list):
        print(f'generating {k}')
        try:
            articulated_count = 0

            with open(functional_part_mapping_dir / f"functional_part_mapping_{k}.pt", "rb") as f:
                functional_part_mapping = torch.load(f)
                
            with open(functional_part_mapping_dir / f"functional_triangles_{k}.pt", "rb") as f:
                functional_structure_triangles = torch.load(f)
                
            all_part_ids = sorted([int(x) for x in functional_part_mapping.keys() if x != 'shape'])
            
            if all(part_id == -1 for part_id in all_part_ids):
                articulated = False
                output_dir_image = output_dir_image_rigid
                output_dir_mesh = output_dir_mesh_rigid
            else:
                articulated = True
                articulated_count += 1
                output_dir_image = output_dir_image_articulated
                output_dir_mesh = output_dir_mesh_articulated
            
            sequence_in_shape_structure = torch.tensor(functional_part_mapping['shape']).unsqueeze(0).to(device) # 72 * sem shape (whole shape) + 2
            bbox_vertices_s, bbox_faces_s, bbox_face_labels_s, shape_geo_feat, arti_exist, joint_types, joint_locs, joint_oris = sequencer_structure.decode(sequence_in_shape_structure[0], decoder_structure, return_semantics=True, return_geo_feat=True, return_articulation_info=True, semantic_retrieval_array=torch.tensor(dataset.semantic_retrieval_array), semantic_feature_decipher=dataset.semantic_feature_decipher, return_token_dict=False)
            plot_vertices_and_faces_withfacelabels(bbox_vertices_s, bbox_faces_s, bbox_face_labels_s, output_dir_image / f"{start_idx + k:06d}_shape_structure.jpg")
            
            junction_pool = {'triangles': [], 'tokens': []}
                
            generated_parts = {}
            whole_verts = []
            whole_faces = []
            whole_face_labels = []
            
            generated_part_ids = set()
            
            sorted_part_ids = get_functional_part_ordering(config.ft_category, functional_structure_triangles['shape']['vertices'], functional_structure_triangles['shape']['faces'], functional_structure_triangles['shape']['face_labels'])
            
            valids = []
            for selected_part_id in sorted_part_ids:

                generated_parts[selected_part_id] = {}
                sequence_in_part_structure = torch.tensor(functional_part_mapping[selected_part_id]).unsqueeze(0).to(device) # 72 * sem part (same functional part) + 2
                sequence_in_shape_structure = torch.tensor(functional_part_mapping['shape']).unsqueeze(0).to(device) # 72 * sem shape (whole shape) + 2
                
                # part structure are used for querying the junction pool
                bbox_vertices, bbox_faces, bbox_face_labels = sequencer_structure.decode(sequence_in_part_structure[0], decoder_structure, return_semantics=True, semantic_retrieval_array=torch.tensor(dataset.semantic_retrieval_array), semantic_feature_decipher=dataset.semantic_feature_decipher)
                
                # plot_vertices_and_faces_withfacelabels(bbox_vertices, bbox_faces, bbox_face_labels, output_dir_image_parts / f"{start_idx + k:06d}_bbox_{selected_part_id}.jpg")

                # just a start token for the mesh sequence
                soup_sequence, face_in_idx, face_out_idx = torch.tensor([[0]]).to(device), torch.tensor([[0]]).to(device), torch.tensor([[0]]).to(device)
                
                if not articulated:
                    print(f'{k} {selected_part_id} is a rigid object - no junction')
                    junction_existance = torch.tensor([[0]]).to(device)
                    soup_sequence_junction = torch.tensor([[0]]).to(device)
                    selected_junction_triangles = []
                else:
                    part_bbox_mesh = trimesh.Trimesh(vertices=bbox_vertices, faces=bbox_faces, process=False) # z-up
                    if len(junction_pool['triangles']) == 0:
                        junction_existance = torch.tensor([[0]]).to(device)
                        soup_sequence_junction = torch.tensor([[0]]).to(device)
                        selected_junction_triangles = []
                    else:
                        selected_junction_triangles, junction_tokens = query_junction_from_pool(part_bbox_mesh, junction_pool, distance_threshold=0.05)
                        soup_sequence_junction = torch.tensor(junction_tokens, dtype=torch.long).unsqueeze(0).to(device)
                        junction_existance = torch.tensor([[1]]).to(device)
                        
                        if len(junction_tokens) == 0:
                            junction_existance = torch.tensor([[0]]).to(device)
                            soup_sequence_junction = torch.tensor([[0]]).to(device)

                if mode == "topp":
                    y = model.generate(
                        soup_sequence, 
                        face_in_idx, 
                        face_out_idx, 
                        sequencer, 
                        config.max_val_tokens - soup_sequence_junction.shape[1] - sequence_in_part_structure.shape[1],
                        temperature=config.temperature, top_k=config.top_k_tokens, top_p=config.top_p,
                        use_kv_cache=False,
                        structure_tokenizer=sequencer_structure,
                        part_junction_sequence=soup_sequence_junction,
                        part_junction_existance=junction_existance, 
                        part_structure_sequence=sequence_in_part_structure, 
                        shape_structure_sequence=sequence_in_shape_structure, 
                    )
                elif mode == "beam":
                    y = model.generate_with_beamsearch(
                        soup_sequence,
                        face_in_idx,
                        face_out_idx,
                        sequencer,
                        config.max_val_tokens - soup_sequence_junction.shape[1] - sequence_in_part_structure.shape[1],
                        use_kv_cache=True,
                        beam_width=6,
                        structure_tokenizer=sequencer_structure,
                        part_junction_sequence=soup_sequence_junction,
                        part_junction_existance=junction_existance, 
                        part_structure_sequence=sequence_in_part_structure, 
                        shape_structure_sequence=sequence_in_shape_structure,
                    )
                    
                if y is None:
                    print(f"Failed to generate part {selected_part_id}")
                    continue

                generated_part_ids.add(selected_part_id)

                gen_vertices, gen_faces, gen_juncs = sequencer.decode(y[0], decoder)
                
                mesh_out_dir = output_dir_mesh / f"{start_idx + k:06d}"
                mesh_out_dir.mkdir(exist_ok=True, parents=True)

                # if config.ft_category == "03001627" and selected_part_id == 0:
                #     num_junc = int(0.15 * len(gen_juncs))
                #     gen_juncs[-num_junc:] = [1] * num_junc
                junction_triangles = []
                junction_tokens = []
                for face_idx, is_junction in enumerate(gen_juncs):
                    if is_junction:
                        junction_triangles.append(np.array(gen_vertices[gen_faces[face_idx]]))
                        start_token = face_idx * 6 + 1 # Assuming 6 tokens per face
                        end_token = start_token + 6
                        junction_tokens.extend(y[0][start_token:end_token].tolist())

                junction_pool['triangles'].extend(junction_triangles)
                junction_pool['tokens'].extend(junction_tokens)

                mesh = trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False, validate=False, maintain_order=True)
                
                mesh.export(mesh_out_dir / f"{selected_part_id}.obj")
                meshlab_proc(
                    mesh_out_dir / f"{selected_part_id}.obj", mesh_out_dir / f"{selected_part_id}.obj"
                )

                loaded_mesh = trimesh.load_mesh(mesh_out_dir / f"{selected_part_id}.obj")
                
                valid_gen = naive_gen_test(loaded_mesh, part_bbox_mesh, 0.02)
                valids.append(valid_gen)

                gen_face_labels = [selected_part_id] * loaded_mesh.faces.shape[0]
                
                process_part = (mesh_out_dir / f"{selected_part_id}.obj").read_text()
                part_verts = read_vertices(process_part)
                part_faces = read_faces(process_part)
                whole_faces.extend((np.asarray(f) + len(whole_verts)).tolist() for f in part_faces)
                whole_verts.extend(part_verts)
                whole_face_labels.extend(gen_face_labels)

            valid = valids.count(True) == len(valids)
            if set(all_part_ids) != generated_part_ids:
                print(f"WARNING: misaligned number of parts {start_idx + k:06d}")
                valid = False
            
            whole_verts = np.array(whole_verts)
            # if config.ft_category == "03001627" and not pnet.classifier_guided_filter(mesh, config.ft_category):
            #     print('failed pointnet filter')
            #     valid = False
            if not valid:
                print(f"Deleting mesh {mesh_out_dir}")
                os.system(f"rm -rf {mesh_out_dir}")
                print("Deleting images")
                os.system(f"rm -rf {output_dir_image}/{start_idx + k:06d}_*" )
            
            else:
                k = k + 1
                colors = {}
                for label in whole_face_labels:
                    if label not in colors:
                        colors[label] = np.random.rand(3)
                plot_vertices_and_faces_withfacelabels(whole_verts, whole_faces, whole_face_labels, output_dir_image / f"{start_idx + k:06d}.jpg", colors)
            
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load(Path(sys.argv[1]).parents[1] / "config.yaml")
    cfg.resume = sys.argv[1]
    cfg.padding = 0.0
    cfg.num_val_samples = int(sys.argv[4])
    cfg.sequence_stride = cfg.block_size
    cfg.top_p = 0.95
    cfg.temperature = 1.0
    cfg.max_val_tokens = 4500
    cfg.low_augment = True
    if cfg.all_parts_per_epoch:
        print("Not using all parts per epoch during inference")
        cfg.all_parts_per_epoch = False
    cfg.permute_articulated_part = False
    
    # NOTE: need to set correct path to the data and checkpoints
    # additional config for optional visualization
    cfg.vis_box = False
    cfg.vis_junctions = False
    
    main(cfg, sys.argv[2], start_idx=0)

