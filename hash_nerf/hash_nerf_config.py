"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
HASH-NERF configuration file.

"""
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from hash_nerf.data.hash_nerf_dataparser import HashDataParserConfig
from hash_nerf.data.hash_dataset import HashDataset

from hash_nerf.hash_nerf import HashNerfModelConfig


hash_nerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="hash-nerf",  
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=30_000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[HashDataset],
                dataparser=HashDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=HashNerfModelConfig(
                eval_num_rays_per_chunk=4096,
            ),
        ),
        optimizers={
           
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-6),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30_000),
            },
            
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Hash NeRF",
)