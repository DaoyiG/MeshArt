import os
import signal
import sys
import traceback
from pathlib import Path
from random import randint
import datetime

import torch
import wandb
import randomname
from pytorch_lightning.strategies.ddp import DDPStrategy

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from vector_quantize_pytorch import ResidualVQ

from model.pointnet import GraphEncoder
from model.resnet1d import resnet34_decoder
from util.filesystem_logger import FilesystemLogger
from util.misc import get_parameters_from_state_dict

def print_traceback_handler(sig, _frame):
    print(f'Received signal {sig}')
    bt = ''.join(traceback.format_stack())
    print(f'Requested stack trace:\n{bt}')


def quit_handler(sig, frame):
    print(f'Received signal {sig}, quitting.')
    sys.exit(1)


def register_debug_signal_handlers(sig=signal.SIGUSR1, handler=print_traceback_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def register_quit_signal_handlers(sig=signal.SIGUSR2, handler=quit_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%m%d%H%M')}_{name}_{config.experiment}_{randomname.get_name()}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)


    seed_everything(1337 + config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # type: ignore # allow tf32 on cudnn
    torch.multiprocessing.set_sharing_strategy('file_system')  # possible fix for the "OSError: too many files" exception

    register_debug_signal_handlers()
    register_quit_signal_handlers()

    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)

    # use wandb logger instead
    if config.logger == 'wandb':
        logger = WandbLogger(project=f'{name}{config.suffix}', name=config.experiment, id=None, settings=wandb.Settings(start_method='thread'), config=dict(config)) # for finetune
    else:
        logger = TensorBoardLogger(name='tb', save_dir=(Path("runs") / config.experiment))

    checkpoint_callback = ModelCheckpoint(
        dirpath=(Path(config.save_root) / config.experiment / "checkpoints"),
        verbose=False,
        every_n_epochs=config.save_epoch,
        filename='{epoch:02d}-{global_step}',
        auto_insert_metric_name=False,
    )

    gpu_count = torch.cuda.device_count()

    precision = 'bf16' if torch.cuda.is_bf16_supported() else 16
    precision = 32

    if gpu_count > 1:
        trainer = Trainer(
            accelerator='gpu',
            strategy=DDPStrategy(find_unused_parameters=True),
            num_nodes=1,
            precision=precision,
            devices=gpu_count,
            num_sanity_val_steps=config.sanity_steps,
            max_epochs=config.max_epoch,
            limit_val_batches=config.val_check_percent,
            callbacks=[checkpoint_callback],
            val_check_interval=float(min(config.val_check_interval, 1)),
            check_val_every_n_epoch=max(1, config.val_check_interval),
            logger=logger,
            deterministic=False,
            benchmark=True,
        )
    elif gpu_count == 1:
        trainer = Trainer(
            devices=[0],
            accelerator='gpu',
            precision=precision,
            strategy=DDPStrategy(find_unused_parameters=True),
            num_sanity_val_steps=config.sanity_steps,
            max_epochs=config.max_epoch,
            limit_val_batches=config.val_check_percent,
            callbacks=[checkpoint_callback],
            val_check_interval=float(min(config.val_check_interval, 1)),
            check_val_every_n_epoch=max(1, config.val_check_interval),
            logger=logger,
            deterministic=False,
            benchmark=True,
        )
    else:
        trainer = Trainer(
            accelerator='cpu',
            precision=precision,
            num_sanity_val_steps=config.sanity_steps,
            max_epochs=config.max_epoch,
            limit_val_batches=config.val_check_percent,
            callbacks=[checkpoint_callback],
            val_check_interval=float(min(config.val_check_interval, 1)),
            check_val_every_n_epoch=max(1, config.val_check_interval),
            logger=logger,
            deterministic=False,
            benchmark=True,
        )
    return trainer


def step(opt, modules):
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1)  # type: ignore
    opt.step()


def create_conv_batch(encoded_features, batch, batch_size, device):
    conv_input, conv_mask = [], []
    max_sequence_length = 0
    for k in range(batch_size):
        features = encoded_features[batch == k, :].T.contiguous().unsqueeze(0)
        max_sequence_length = max(max_sequence_length, features.shape[2])
        conv_input.append(features)
        conv_mask.append(torch.ones([features.shape[2]], device=device, dtype=torch.bool))
    for k in range(batch_size):
        conv_input[k] = torch.nn.functional.pad(conv_input[k], (0, max_sequence_length - conv_input[k].shape[2]), 'replicate')
        conv_mask[k] = torch.nn.functional.pad(conv_mask[k], (0, max_sequence_length - conv_mask[k].shape[0]), 'constant', False)
    conv_input = torch.cat(conv_input, dim=0)
    conv_mask = torch.cat(conv_mask, dim=0)
    return conv_input, conv_mask


def get_rvqvae_v0_all(config, resume):
    encoder, pre_quant, post_quant, vq = get_rvqvae_v0_encoder_vq(config, resume)
    decoder = get_rvqvae_v0_decoder(config, resume)
    return encoder, decoder, pre_quant, post_quant, vq


def get_rvqvae_v0_encoder_vq(config, resume):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, order_invariant=config.order_invariant)
    pre_quant = torch.nn.Linear(512, config.embed_dim)
    post_quant = torch.nn.Linear(config.embed_dim, 512)

    encoder.load_state_dict(get_parameters_from_state_dict(state_dict, "encoder"))
    pre_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "pre_quant"))
    post_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "post_quant"))

    vq = ResidualVQ(
        dim=config.embed_dim,
        codebook_size=config.n_embed,  # codebook size
        num_quantizers=config.embed_levels,
        commitment_weight=config.embed_loss_weight,  # the weight on the commitment loss
        stochastic_sample_codes=True,
        sample_codebook_temp=0.1,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook=config.embed_share,
        decay=config.code_decay,
    )
    vq.load_state_dict(get_parameters_from_state_dict(state_dict, "vq"))
    return encoder, pre_quant, post_quant, vq


def get_rvqvae_v0_decoder(config, resume, device=torch.device("cpu")):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output)
    decoder.load_state_dict(get_parameters_from_state_dict(state_dict, "decoder"))
    decoder = decoder.to(device).eval()
    return decoder


def get_rvqvae_v0junc_decoder(config, resume, device=torch.device("cpu")):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output, pred_junc=True)
    decoder.load_state_dict(get_parameters_from_state_dict(state_dict, "decoder"))
    decoder = decoder.to(device).eval()
    return decoder

def get_rvqvae_v1_encoder_vq(config, resume):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576, order_invariant=config.order_invariant)
    pre_quant = torch.nn.Linear(192, config.embed_dim)
    post_quant = torch.nn.Linear(config.embed_dim * 3, 512)

    encoder.load_state_dict(get_parameters_from_state_dict(state_dict, "encoder"))
    pre_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "pre_quant"))
    post_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "post_quant"))

    vq = ResidualVQ(
        dim=config.embed_dim,
        codebook_size=config.n_embed,  # codebook size
        num_quantizers=config.embed_levels,
        commitment_weight=config.embed_loss_weight,  # the weight on the commitment loss
        stochastic_sample_codes=True,
        sample_codebook_temp=0.1,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook=config.embed_share,
        decay=config.code_decay,
    )
    vq.load_state_dict(get_parameters_from_state_dict(state_dict, "vq"))
    return encoder, pre_quant, post_quant, vq

def get_rvqvae_v4_encoder_vq(config, resume):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576, order_invariant=config.order_invariant, semantic_features=True, geometry_features=True, articulation_features=True, pc_head=False)
    pre_quant = torch.nn.Linear(192, config.embed_dim)
    post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
    
    encoder.load_state_dict(get_parameters_from_state_dict(state_dict, "encoder"))
    pre_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "pre_quant"))
    post_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "post_quant"))
    
    vq = ResidualVQ(
        dim=config.embed_dim,
        codebook_size=config.n_embed,  # codebook size
        num_quantizers=config.embed_levels,
        commitment_weight=config.embed_loss_weight,  # the weight on the commitment loss
        stochastic_sample_codes=True,
        sample_codebook_temp=0.1,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook=config.embed_share,
        decay=config.code_decay,
    )
    vq.load_state_dict(get_parameters_from_state_dict(state_dict, "vq"))
    return encoder, pre_quant, post_quant, vq


def get_rvqvae_v4_decoder(config, resume, device=torch.device("cpu")):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output, sem_head=True, geo_head=True, arti_head=True, pc_head=False, ft_category=config.ft_category)
    decoder.load_state_dict(get_parameters_from_state_dict(state_dict, "decoder"))
    decoder = decoder.to(device).eval()
    return decoder

def get_rvqvae_v4_encoder_vq_decoder(config, resume):
    state_dict = torch.load(resume, map_location="cpu")["state_dict"]
    encoder = GraphEncoder(no_max_pool=config.g_no_max_pool, aggr=config.g_aggr, graph_conv=config.graph_conv, use_point_features=config.use_point_feats, output_dim=576, order_invariant=config.order_invariant, semantic_features=True, geometry_features=True, articulation_features=True, pc_head=False)
    pre_quant = torch.nn.Linear(192, config.embed_dim)
    post_quant = torch.nn.Linear(config.embed_dim * 3, 512)
    decoder = resnet34_decoder(512, config.num_tokens - 2, config.ce_output, sem_head=True, geo_head=True, arti_head=True, pc_head=False)
    
    encoder.load_state_dict(get_parameters_from_state_dict(state_dict, "encoder"))
    pre_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "pre_quant"))
    post_quant.load_state_dict(get_parameters_from_state_dict(state_dict, "post_quant"))
    decoder.load_state_dict(get_parameters_from_state_dict(state_dict, "decoder"))
    
    
    vq = ResidualVQ(
        dim=config.embed_dim,
        codebook_size=config.n_embed,  # codebook size
        num_quantizers=config.embed_levels,
        commitment_weight=config.embed_loss_weight,  # the weight on the commitment loss
        stochastic_sample_codes=True,
        sample_codebook_temp=0.1,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
        shared_codebook=config.embed_share,
        decay=config.code_decay,
    )
    vq.load_state_dict(get_parameters_from_state_dict(state_dict, "vq"))
    return encoder, pre_quant, post_quant, vq, decoder

