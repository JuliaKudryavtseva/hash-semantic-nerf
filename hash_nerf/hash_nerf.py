"""
Implementation of Hash Nerf.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type
from collections import defaultdict

import os
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity as cs
import pickle
import random
import colorsys

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler 
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    RGBRenderer,
    DepthRenderer,
) 
from nerfstudio.model_components.scene_colliders import NearFarCollider 
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.viewer.server.viewer_elements import ViewerText

from hash_nerf.hash_nerf_fieldheadname import HashFieldHeadNames
from hash_nerf.hash_nerf_fields import HashNerfField
from hash_nerf.hash_nerf_renderer import HashRenderer

from tqdm import tqdm
import gc


def random_colors(N):
    random_color=np.random.choice(range(255), size=(N, 3))
    random_color[0] = np.zeros(3)
    return torch.tensor(random_color)

@dataclass
class HashNerfModelConfig(ModelConfig):
    """HashNerfModel Config"""

    _target: Type = field(
        default_factory=lambda: HashNerfModel
    )  
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.01
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    sem_loss_weight: float = 1
    


class HashNerfModel(Model):
    """Sematic NeRF model 

    Args:
        config: sam_nerf configuration to instantiate model
    """

    config: HashNerfModelConfig
    field: HashNerfField

    def __init__(self, config: HashNerfModelConfig, metadata: Dict, **kwargs) -> None:
        
        super().__init__(config=config, **kwargs)

        # data_path = os.environ['DATA_PATH']
        # self.labels_n_dims = int(np.load(f"data/{data_path}/consistent_masks/number_class.npy") + 1)
        # self.colors = random_colors(self.labels_n_dims).cuda()

        # with open(f"data/{data_path}/consistent_masks/label_map_colors.pkl", 'rb') as h:
        #     self.colors = pickle.load(h)

        # # DICT WITH SAM EMBEDDINGS

        # data_path = os.environ['DATA_PATH']
        # path_masks = f"data/{data_path}/consistent_masks/ready_masks/" #figurines
        # self.pickle_labels = {}
        # print('Load dataset')
        # for fname in tqdm(os.listdir(path_masks)):
        #     with open(path_masks + fname, 'rb') as h:
        #         label_emb_dict = pickle.load(h)
        #     self.pickle_labels[int(fname.split("_")[1])] = label_emb_dict

        # # DICT WITH SAM EMBEDDINGS

        


    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # # CREATE SAM MODEL
        # self.positive_input = ViewerText("Text Positives", "", cb_hook=self.gui_cb)
        # # CREATE SAM MODEL
        data_path = os.environ['DATA_PATH']
        self.class_weights = torch.tensor(np.load(f"data/{data_path}/consistent_masks/weights.npy")) 

        # self.class_weights[0] = 0.1
        self.labels_n_dims = int(np.load(f"data/{data_path}/consistent_masks/number_class.npy"))+1
        self.colors = random_colors(self.labels_n_dims).cuda()
        

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = HashNerfField(
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=0 if self.config.use_appearance_embedding else 32,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size, 
            max_res=self.config.max_res, # 2048 Maximum resolution of the hashmap for the base mlp 
            spatial_distortion=scene_contraction,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
            
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_semantic = HashRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # class_weights=torch.ones(self.labels_n_dims)
        # class_weights[0]=0.1
        # weight=self.class_weights

        # self.class_weights[self.class_weights > 0.09] = 1
        self.semantic_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.max_raw_relevancy = 0
        self.min_raw_relevancy = 1
        self.positives = 1


    # # TOKENIZER FOR TEXT PROMPT
    # def gui_cb(self,element):
    #     self.set_positives(element.value.split(";"))

    # def set_positives(self, text):
    #     self.positives = int(text[0])
        

    # # TOKENIZER FOR TEXT PROMPT 

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.config.render_step_size,
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups


    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        # The ray_indices contains the indices of the rays that each sample belongs to.

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]

        weights = weights[..., None]
        depth = self.renderer_depth(weights=weights, 
                                    ray_samples=ray_samples, 
                                    ray_indices=ray_indices,
                                    num_rays=num_rays,
                                    )



        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "num_samples_per_ray": packed_info[:, 1],
            "depth": depth
        }

        # label OUTPUT
        if self.training:
            outputs["pred_labels"] = self.renderer_semantic(
                    embeds=field_outputs[HashFieldHeadNames.pred_labels], 
                    weights=weights.detach(), 
                    ray_indices=ray_indices, 
                    num_rays=num_rays
                )

        
        if not self.training:
            with torch.no_grad(): 
                outputs["pred_labels"] = self.get_sem_vis(
                    field_outputs,
                    ray_samples,
                    weights,
                    ray_indices,
                    num_rays                 
                ) # B x 5611

        return outputs
        # label PREDICITON   

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}

        # ================================
        image = batch["image"][..., :3].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)
        loss_dict = {"rgb_loss": rgb_loss}

        # ================================


        batch['semantics'] = torch.nn.functional.one_hot(batch['semantics'].long(), num_classes=self.labels_n_dims)


        # batch["true_hash_labels"] = []
        # for ind_frame, ind_label in zip(batch['frame_number'], batch['labels_map']):
        #     frame_labels = self.pickle_labels[int(ind_frame)]
        #     one_hot_labels = torch.zeros(self.labels_dim, dtype=int)

        #     if int(ind_label) in frame_labels.keys():
        #         labels = frame_labels[int(ind_label)]
        #         # one_hot_labels[:len(labels)] = torch.tensor(labels)
        #         one_hot_labels[labels] = 1
        #     else:
        #         one_hot_labels[0]= 1 
            
        #     batch["true_hash_labels"].append(one_hot_labels)

        # batch["true_hash_labels"] = torch.tensor(np.array(batch['true_hash_labels'])).cuda().float()

        
        

        if self.training:
            # if outputs["pred_labels"].isnan().sum().item(): 
            #     loss = 1e-6
            #     print('=== Outputs with NANs === ')
            # else:
            loss_semantics = self.semantic_loss(outputs["pred_labels"], batch["semantics"].float())
            loss_dict["sam_loss"] = self.config.sem_loss_weight*loss_semantics.nanmean()

        del batch["semantics"]
        # del outputs["pred_labels"]
        torch.cuda.empty_cache()
        
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
        }

        return metrics_dict, images_dict


    # ---sam visualisation---
    
    def get_sem_vis(self, field_outputs, ray_samples, weights, ray_indices, num_rays):
        field_outputs = self.field(ray_samples)

        sem_output = self.renderer_semantic(
                embeds=field_outputs[HashFieldHeadNames.pred_labels], 
                weights=weights.detach(),
                ray_indices=ray_indices, 
                num_rays=num_rays
            ) #  [B, 853]


        semantics_output = torch.argmax(sem_output, dim=1) # B
        del sem_output
        gc.collect()
        torch.cuda.empty_cache()


        train = True
        if train == True:
            return self.colors[semantics_output] 
        else:
            prediction = torch.zeros((semantics_output.shape[0], 3))
            prediction[:, 0] = semantics_output
            return prediction

        

        
        



        

    # MODEL PREDICITON
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        # Takes in camera parameters and computes the output of the model.
        # camera_ray_bundle: ray bundle to calculate outputs over
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():  
                outputs_lists[output_name].append(output)

        # SAM RELEVANCY
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)
            
        return outputs