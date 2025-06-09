import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import random
import omegaconf
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from easydict import EasyDict
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import numpy as np
import torch

from dataset.quantized_soup import QuantizedSoupTripletsCreator
from dataset.quantized_soup_structure import QuantizedSoupTripletsCreator as QuantizedSoupTripletsCreatorStructure
from dataset.triangles_geometry import TriangleNodesWithFacesAndSequenceIndices, TriangleNodesWithFacesDataloader
from model.transformer_geometry import QuantSoupTransformer as QuantSoupTransformerNoEmbed
from trainer import create_trainer, step, get_rvqvae_v0_decoder, get_rvqvae_v4_decoder
from util.misc import accuracy
from util.visualization import plot_vertices_and_faces, plot_vertices_and_faces_withfacelabels
from util.misc import get_parameters_from_state_dict

class QuantSoupModelTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.vq_cfg_structure = omegaconf.OmegaConf.load(Path(config.vq_resume_structure).parents[1] / "config.yaml")
        self.save_hyperparameters()
        self.train_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', config.scale_augment, config.shift_augment, config.ft_category, joint_augment=config.joint_augment, junction_augment=config.junction_augment)
        self.val_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'val', config.scale_augment_val, False, config.ft_category, joint_augment=config.joint_augment, junction_augment=config.junction_augment_val)
        print("Dataset Lengths:", len(self.train_dataset), len(self.val_dataset))
        print("Batch Size:", self.config.batch_size)
        print("Dataloader Lengths:", len(self.train_dataset) // self.config.batch_size, len(self.val_dataset) // self.config.batch_size)
        model_cfg = get_qsoup_model_config(config, self.vq_cfg.embed_levels)
        self.model = QuantSoupTransformerNoEmbed(model_cfg, self.vq_cfg)
        self.sequencer = QuantizedSoupTripletsCreator(self.config, self.vq_cfg)
        self.sequencer.freeze_vq()
        self.sequencer_structure = QuantizedSoupTripletsCreatorStructure(self.config, self.vq_cfg_structure)
        self.sequencer_structure.freeze_vq()
        
        self.output_dir_image = Path(f'runs/{self.config.experiment}/image')
        self.output_dir_image.mkdir(exist_ok=True, parents=True)
        self.output_dir_image_train = Path(f'runs/{self.config.experiment}/image_train')
        self.output_dir_image_train.mkdir(exist_ok=True, parents=True)
        
        if self.config.ft_resume is not None:
            pretrained_state_dict = get_parameters_from_state_dict(torch.load(self.config.ft_resume, map_location='cpu')['state_dict'], "model")
            for name in self.model.state_dict():
                if name in pretrained_state_dict:
                    self.model.state_dict()[name].copy_(pretrained_state_dict[name])
            missing_keys = [name for name in self.model.state_dict() if name not in pretrained_state_dict]
            print('Missing keys:', missing_keys)
        
        self.automatic_optimization = False
        

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.config.weight_decay, self.config.lr,
            (self.config.beta1, self.config.beta2), 'cuda'
        )
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / (self.config.batch_size))
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()
        if self.config.force_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.force_lr
        
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data['part'].x, data['part'].edge_index, data['part'].batch, data['part'].faces, data['part'].num_vertices.sum(), data['part'].js)
        sequence_in_junction, _, _, _ = self.sequencer(data['part_junction'].x, data['part_junction'].edge_index, data['part_junction'].batch, data['part_junction'].faces, data['part_junction'].num_vertices.sum(), data['part_junction'].js)
        sequence_in_part_structure, _, _, _ = self.sequencer_structure(data['part_structure'].x, data['part_structure'].edge_index, data['part_structure'].batch, data['part_structure'].faces, data['part_structure'].num_vertices.sum(), data['part_structure'].js, semantic_features=data['part_structure'].semantic_features, geometry_features=data['part_structure'].geometry_features, articulation_features=data['part_structure'].articulation_features)
        sequence_in_shape_structure, _, _, _ = self.sequencer_structure(data['shape_structure'].x, data['shape_structure'].edge_index, data['shape_structure'].batch, data['shape_structure'].faces, data['shape_structure'].num_vertices.sum(), data['shape_structure'].js, semantic_features=data['shape_structure'].semantic_features, geometry_features=data['shape_structure'].geometry_features, articulation_features=data['shape_structure'].articulation_features)
        
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out, part_junction_sequence=sequence_in_junction, part_junction_existance=data['part_junction'].junction_existance, part_structure_sequence=sequence_in_part_structure, shape_structure_sequence=sequence_in_shape_structure, structure_tokenizer=self.sequencer_structure)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        self.log("train/ce_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self.log("train/acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)

        loss = loss / self.config.gradient_accumulation_steps
        self.manual_backward(loss)

        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.model])
            optimizer.zero_grad(set_to_none=True)
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)  # type: ignore
    
    def find_unused_parameters(self):
        unused_params = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                unused_params.append(name)
        return unused_params

    
    def validation_step(self, data, batch_idx):
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data['part'].x, data['part'].edge_index, data['part'].batch, data['part'].faces, data['part'].num_vertices.sum(), data['part'].js)
        sequence_in_junction, _, _, _ = self.sequencer(data['part_junction'].x, data['part_junction'].edge_index, data['part_junction'].batch, data['part_junction'].faces, data['part_junction'].num_vertices.sum(), data['part_junction'].js)
        sequence_in_part_structure, _, _, _ = self.sequencer_structure(data['part_structure'].x, data['part_structure'].edge_index, data['part_structure'].batch, data['part_structure'].faces, data['part_structure'].num_vertices.sum(), data['part_structure'].js, semantic_features=data['part_structure'].semantic_features, geometry_features=data['part_structure'].geometry_features, articulation_features=data['part_structure'].articulation_features)
        sequence_in_shape_structure, _, _, _ = self.sequencer_structure(data['shape_structure'].x, data['shape_structure'].edge_index, data['shape_structure'].batch, data['shape_structure'].faces, data['shape_structure'].num_vertices.sum(), data['shape_structure'].js, semantic_features=data['shape_structure'].semantic_features, geometry_features=data['shape_structure'].geometry_features, articulation_features=data['shape_structure'].articulation_features)
        
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out, part_junction_sequence=sequence_in_junction, part_junction_existance=data['part_junction'].junction_existance, part_structure_sequence=sequence_in_part_structure, shape_structure_sequence=sequence_in_shape_structure, structure_tokenizer=self.sequencer_structure)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        if not torch.isnan(loss).any():
            self.log("val/ce_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc).any():
            self.log("val/acc", acc.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self):
        decoder = get_rvqvae_v0_decoder(self.vq_cfg, self.config.vq_resume, self.device)
        decoder_structure = get_rvqvae_v4_decoder(self.vq_cfg_structure, self.config.vq_resume_structure, self.device)
        for k in range(self.config.num_val_samples):
            data = self.val_dataset.get(random.randint(0, len(self.val_dataset) - 1))
            soup_sequence, face_in_idx, face_out_idx, target = self.sequencer.get_completion_sequence(
                data['part'].x.to(self.device),
                data['part'].edge_index.to(self.device),
                data['part'].faces.to(self.device),
                data['part'].num_vertices,
                12
            )
            
            soup_sequence_junction, _, _, _ = self.sequencer.get_completion_sequence(
                data['part_junction'].x.to(self.device),
                data['part_junction'].edge_index.to(self.device),
                data['part_junction'].faces.to(self.device),
                data['part_junction'].num_vertices,
                -1
            )
            
            batch_part_structure = []
            length_part_structure = data['part_structure'].feature_length_structure
            batch_part_structure.extend([0] * length_part_structure)
            batch_part_structure = torch.tensor(batch_part_structure).to(self.device)
                
            sequence_in_part_structure, _, _, _ = self.sequencer_structure(data['part_structure'].x.to(self.device), 
                                                                    data['part_structure'].edge_index.to(self.device), 
                                                                    batch_part_structure, 
                                                                    data['part_structure'].faces.to(self.device),
                                                                    data['part_structure'].num_vertices, 
                                                                    torch.zeros([1],device=self.device).long(), 
                                                                    semantic_features=data['part_structure'].semantic_features.to(self.device),
                                                                    geometry_features=data['part_structure'].geometry_features.to(self.device), 
                                                                    articulation_features=data['part_structure'].articulation_features.to(self.device)
                                                                    )
            
            batch_shape_structure = []
            length_structure = data['shape_structure'].feature_length_structure
            batch_shape_structure.extend([0] * length_structure)
            batch_shape_structure = torch.tensor(batch_shape_structure).to(self.device)
                
            sequence_in_shape_structure, _, _, _ = self.sequencer_structure(data['shape_structure'].x.to(self.device), 
                                                                    data['shape_structure'].edge_index.to(self.device), 
                                                                    batch_shape_structure, 
                                                                    data['shape_structure'].faces.to(self.device),
                                                                    data['shape_structure'].num_vertices, 
                                                                    torch.zeros([1],device=self.device).long(), 
                                                                    semantic_features=data['shape_structure'].semantic_features.to(self.device), 
                                                                    geometry_features=data['shape_structure'].geometry_features.to(self.device), 
                                                                    articulation_features=data['shape_structure'].articulation_features.to(self.device)
                                                                    )
            y = self.model.generate(
                soup_sequence, face_in_idx, face_out_idx, self.sequencer, 
                self.config.max_val_tokens-data['part_junction'].x.shape[0]*6-data['part_structure'].x.shape[0]*6,
                temperature=self.config.temperature, top_k=self.config.top_k_tokens, top_p=self.config.top_p,
                use_kv_cache=False,
                structure_tokenizer=self.sequencer_structure,
                part_junction_sequence=soup_sequence_junction,
                part_junction_existance=data['part_junction'].junction_existance.reshape(1, -1).to(self.device), 
                part_structure_sequence=sequence_in_part_structure, 
                shape_structure_sequence=sequence_in_shape_structure, 
            )
            if y is None:
                continue
            
            gen_vertices, gen_faces = self.sequencer.decode(y[0], decoder)
            triangles = data['part'].x[:, :9]
            gt_mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles.reshape(-1, 3, 3).cpu()), process=False)
            gt_vertices, gt_faces = gt_mesh.vertices, gt_mesh.faces
            plot_vertices_and_faces(gen_vertices, gen_faces, self.output_dir_image / f"{self.global_step:06d}_{k}.jpg")
            plot_vertices_and_faces(gt_vertices, gt_faces, self.output_dir_image / f"{self.global_step:06d}_{k}_gt.jpg")
            
            
            part_bbox_vertices, part_bbox_faces, part_bbox_face_labels = self.sequencer_structure.decode(sequence_in_part_structure[0], decoder_structure, return_semantics=True, semantic_retrieval_array=torch.tensor(self.val_dataset.semantic_retrieval_array), semantic_feature_decipher=self.val_dataset.semantic_feature_decipher)
            
            bbox_vertices, bbox_faces, bbox_face_labels = self.sequencer_structure.decode(sequence_in_shape_structure[0], decoder_structure, return_semantics=True, semantic_retrieval_array=torch.tensor(self.val_dataset.semantic_retrieval_array), semantic_feature_decipher=self.val_dataset.semantic_feature_decipher)
            plot_vertices_and_faces(bbox_vertices, bbox_faces, self.output_dir_image / f"{self.global_step:06d}_{k}_shapebbox.jpg", 'chocolate')
            
            # combine the generated vertices and faces with bbox
            gen_vertices_withbbox = np.concatenate((gen_vertices, part_bbox_vertices), axis=0)
            gen_faces_withbbox = np.concatenate((gen_faces, part_bbox_faces + gen_vertices.shape[0]), axis=0)
            face_labels = [-1] * gen_faces.shape[0] + [0] * len(part_bbox_face_labels)
            plot_vertices_and_faces_withfacelabels(gen_vertices_withbbox, gen_faces_withbbox, face_labels, self.output_dir_image / f"{self.global_step:06d}_{k}_gen_wbbox.jpg")

    def train_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=self.config.num_workers)

def get_qsoup_model_config(config, vq_embed_levels):
    cfg = EasyDict({
        'block_size': config.block_size,
        'n_embd': config.model.n_embd,
        'dropout': config.model.dropout,
        'n_layer': config.model.n_layer,
        'n_head': config.model.n_head,
        'bias': config.model.bias,
        'finemb_size': vq_embed_levels * 3,
        'foutemb_size': config.block_size * 3,
    })
    return cfg


@hydra.main(config_path='../config', config_name='nanogpt', version_base='1.2')
def main(config):
    trainer = create_trainer("MeshTriSoup", config)
    model = QuantSoupModelTrainer(config)
    trainer.fit(model, ckpt_path=config.resume)

if __name__ == '__main__':
    main()
