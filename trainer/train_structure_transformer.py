import random
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import omegaconf
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from easydict import EasyDict
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dataset.quantized_soup_structure import QuantizedSoupTripletsCreator
from dataset.triangles_structure import TriangleNodesWithFacesAndSequenceIndices, TriangleNodesWithFacesDataloader
from model.transformer_structure import QuantSoupTransformer as QuantSoupTransformerStructure
from trainer import create_trainer, step, get_rvqvae_v4_decoder
from util.misc import accuracy
from util.visualization import plot_vertices_and_faces_withfacelabels, plot_vertices_and_faces_withfacelabels_wjoint, plot_vertices_and_faces_withfacelabels_wjointonpart
from util.misc import get_parameters_from_state_dict
import numpy as np


class QuantSoupModelTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.save_hyperparameters()
        self.train_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', config.scale_augment, config.shift_augment, config.ft_category, config.joint_augment)
        self.val_dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'val', config.scale_augment_val, False, config.ft_category, config.joint_augment_val)
        print("Dataset Lengths:", len(self.train_dataset), len(self.val_dataset))
        print("Dataloader Lengths:", len(self.train_dataset) // self.config.batch_size, len(self.val_dataset) // self.config.batch_size)
        print("Batch Size:", self.config.batch_size)
        model_cfg = get_qsoup_model_config(config, self.vq_cfg.embed_levels)
        self.model = QuantSoupTransformerStructure(model_cfg, self.vq_cfg)
        self.sequencer = QuantizedSoupTripletsCreator(self.config, self.vq_cfg)
        self.sequencer.freeze_vq()
        self.output_dir_image = Path(f'runs/{self.config.experiment}/image')
        self.output_dir_image.mkdir(exist_ok=True, parents=True)
        self.output_dir_image_train = Path(f'runs/{self.config.experiment}/image_train')
        self.output_dir_image_train.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh = Path(f'runs/{self.config.experiment}/mesh')
        self.output_dir_mesh.mkdir(exist_ok=True, parents=True)
        if self.config.ft_resume is not None:
            self.model.load_state_dict(get_parameters_from_state_dict(torch.load(self.config.ft_resume, map_location='cpu')['state_dict'], "model"), strict=False)
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.config.weight_decay, self.config.lr,
            (self.config.beta1, self.config.beta2), 'cuda'
        )
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size * 2)
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        if self.config.force_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.force_lr
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data.x, data.edge_index, data.batch, data.faces, data.num_vertices.sum(), data.js, semantic_features=data.semantic_features, geometry_features=data.geometry_features, articulation_features=data.articulation_features)
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        self.log("train/ce_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        self.log("train/acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=False)
        loss = loss / self.config.gradient_accumulation_steps
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.model])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)  # type: ignore
    
    def validation_step(self, data, batch_idx):
        sequence_in, sequence_out, pfin, pfout = self.sequencer(data.x, data.edge_index, data.batch, data.faces, data.num_vertices.sum(), data.js, semantic_features=data.semantic_features, geometry_features=data.geometry_features, articulation_features=data.articulation_features)
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        if not torch.isnan(loss).any():
            self.log("val/ce_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc).any():
            self.log("val/acc", acc.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self):
        decoder = get_rvqvae_v4_decoder(self.vq_cfg, self.config.vq_resume, self.device)
        for k in range(self.config.num_val_samples):
            data = self.val_dataset.get(k if self.config.overfit else random.randint(0, len(self.val_dataset) - 1))
            
            soup_sequence, face_in_idx, face_out_idx, target = self.sequencer.get_completion_sequence(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                data.faces.to(self.device),
                data.num_vertices,
                12,
                data.semantic_features.to(self.device),
                data.geometry_features.to(self.device),
                data.articulation_features.to(self.device)
            )

            if self.config.overfit:
                y = self.model.generate(
                    soup_sequence, face_in_idx, face_out_idx, self.sequencer, self.config.max_val_tokens,
                    temperature=self.config.temperature, top_k=1, top_p=None,
                    use_kv_cache=False,
                )
            else:
                y = self.model.generate(
                    soup_sequence, face_in_idx, face_out_idx, self.sequencer, self.config.max_val_tokens,
                    temperature=self.config.temperature, top_k=self.config.top_k_tokens, top_p=self.config.top_p,
                    use_kv_cache=False,
                )
            if y is None:
                continue
            try:
                gen_vertices, gen_faces, gen_face_labels, decoded_x_conv_geo, gen_face_arti_exist, gen_face_arti_type, coords_arti_loc, coords_arti_ori = self.sequencer.decode(y[0], decoder, return_semantics=True, return_geo_feat=True, return_articulation_info=True, semantic_retrieval_array=torch.tensor(self.val_dataset.semantic_retrieval_array), semantic_feature_decipher=self.val_dataset.semantic_feature_decipher)
                
                gt_vertices, gt_faces, gt_face_labels, gt_geo, gt_face_arti_exist, gt_face_arti_type, gt_coords_arti_loc, gt_coords_arti_ori = self.sequencer.decode(target[0], decoder, return_semantics=True, return_geo_feat=True, return_articulation_info=True, semantic_retrieval_array=torch.tensor(self.val_dataset.semantic_retrieval_array), semantic_feature_decipher=self.val_dataset.semantic_feature_decipher)
                
                num_face_labels_gen = np.unique(gen_face_labels)
                colors_gen = {v: (random.random(), random.random(), random.random()) for v in (num_face_labels_gen)}
                
                num_face_labels_gt = np.unique(gt_face_labels)
                colors_gt = {v: (random.random(), random.random(), random.random()) for v in (num_face_labels_gt)}
                
                plot_vertices_and_faces_withfacelabels(gen_vertices, gen_faces, gen_face_labels, self.output_dir_image / f"{self.global_step:06d}_{k}_gen.jpg", color=colors_gen)
                plot_vertices_and_faces_withfacelabels(gt_vertices, gt_faces, gt_face_labels, self.output_dir_image / f"{self.global_step:06d}_{k}_gt.jpg", color=colors_gt)
                
                
                plot_vertices_and_faces_withfacelabels_wjoint(gen_vertices, gen_faces, gen_face_labels, coords_arti_loc, coords_arti_ori, gen_face_arti_type, self.output_dir_image / f"{self.global_step:06d}_{k}_gen_joint.jpg", color=colors_gen)
                plot_vertices_and_faces_withfacelabels_wjoint(gt_vertices, gt_faces, gt_face_labels, gt_coords_arti_loc, gt_coords_arti_ori, gt_face_arti_type, self.output_dir_image / f"{self.global_step:06d}_{k}_gt_joint.jpg", color=colors_gt)

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
                plot_vertices_and_faces_withfacelabels_wjointonpart(gen_vertices, gen_faces, bbox_inst_face_labels, new_joint_locs, coords_arti_ori, joint_types, self.output_dir_image / f"{self.global_step:06d}_{k}_jointonpart.jpg", color=bbox_inst_face_colors)

            except Exception as e:
                pass


            
    def train_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=not self.config.overfit, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        return TriangleNodesWithFacesDataloader(self.val_dataset, batch_size=self.config.batch_size, shuffle=not self.config.overfit, drop_last=False, num_workers=self.config.num_workers)


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
