# Autoregressive model for RigAnything, add diffusion loss, and skining weight prediction
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from .diffloss import DiffLoss
from .utils_ar_transformer import (CustomTransformerBlock, _init_weights)

class RigARDiffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Pointcloud encoder
        self.pc_tokenizer = nn.Sequential(
            nn.Linear(
                config.model.pc_tokenizer.in_channels,
                config.model.pc_tokenizer.middle_channels,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.pc_tokenizer.middle_channels,
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.pc_tokenizer.apply(_init_weights)

        # Joint encoder
        self.joint_tokenizer = nn.Sequential(
            nn.Linear(
                config.model.joints_tokenizer.in_channels,
                config.model.joints_tokenizer.middle_channels,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.joints_tokenizer.middle_channels,
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.joint_tokenizer.apply(_init_weights)

        # Joint index position embedding
        self.joint_index_pos_embedding = self.positional_encoding(
            config.model.joints_tokenizer.n_joints, config.model.transformer.d
        ).to(self.device)
        # nn.init.trunc_normal_(self.joint_index_pos_embedding, std=0.02)  # TODO trunc normal??

        # Joint MLP(joint embedding, parent embedding, joint index)
        self.joint_mlp = nn.Sequential(
            nn.Linear(
                config.model.transformer.d * 4,
                config.model.transformer.d * 2,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.joint_mlp.apply(_init_weights)

        # Joint fuse MLP (joint tokens, joint embedding, joint index)
        self.joint_fuse_mlp = nn.Sequential(
            nn.Linear(
                config.model.transformer.d * 3,
                config.model.transformer.d * 2,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.joint_fuse_mlp.apply(_init_weights)

        # Skinning MLP
        self.skinning_mlp = nn.Sequential(
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.transformer.d,
                1,
                bias=False,
            ),
        )
        self.skinning_mlp.apply(_init_weights)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, config.model.transformer.d))

        # Parent decoder
        self.parents_decoder = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d * 2, bias=False),
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                config.model.transformer.d,
                1,
                bias=False,
            ),
        )
        self.parents_decoder.apply(_init_weights)

        self.transformer_input_layernorm = nn.LayerNorm(
            config.model.transformer.d, bias=False
        )
        self.transformer = nn.ModuleList(
            [
                CustomTransformerBlock(
                    config.model.transformer.d,
                    config.model.transformer.d_head,
                    config.model.pc_tokenizer.n_points,
                    config.model.joints_tokenizer.n_joints,
                )
                for _ in range(config.model.transformer.n_layer)
            ]
        )
        self.transformer.apply(_init_weights)

        # Diffuse Loss for joint positions
        self.diffloss = DiffLoss(
            target_channels=3,
            z_channels=self.config.model.transformer.d,
            width=self.config.model.diffusion.w,
            depth=self.config.model.diffusion.d,
            num_sampling_steps=self.config.model.diffusion.num_sampling_steps,
            grad_checkpointing=self.config.model.diffusion.grad_checkpointing,
        )

        # For time-varying setting
        self.current_step = None
        self.start_step = None
        self.max_step = None
        self.config_bak = copy.deepcopy(config)

    def positional_encoding(self, seq_length, d_model):
        pos = torch.arange(seq_length).unsqueeze(1)
        i = torch.arange(d_model).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates

        # Apply sin to even indices and cos to odd indices
        pos_encoding = torch.zeros(seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_encoding

    def get_custom_attn_mask(self, n_pc, n_joints):
        """
        Construct the custom attention mask for the transformer
        #   P P P P | J J J
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # - - - - - | - - -
        # J 1 1 1 1 | 1 0 0
        # J 1 1 1 1 | 1 1 0
        # J 1 1 1 1 | 1 1 1

        Args:
            n_pc (int): number of points
            n_joints (int): number of joints

        Returns:
            torch.Tensor: [n_pc + n_joints, n_pc + n_joints]
        """
        attn_mask = torch.zeros(n_pc + n_joints, n_pc + n_joints)
        attn_mask[:n_pc, :n_pc] = 1
        attn_mask[n_pc:, n_pc:] = torch.tril(torch.ones(n_joints, n_joints))
        attn_mask[n_pc:, :n_pc] = 1
        return attn_mask.to(torch.bool).to(self.device)

    def set_current_step(self, current_step, start_step, max_step):
        self.current_step = current_step
        self.start_step = start_step
        self.max_step = max_step

        # be careful with modifying configs
        plan_to_modify_config = (
            self.config.training.get("warmup_pointsdist", False)
            and self.current_step < 1000
        ) or (
            self.config.training.get("l2_warmup_steps", 0) > 0
            and self.current_step < self.config.training.l2_warmup_steps
        )
        if plan_to_modify_config:
            # always use the self.config_bak as the starting point for modification
            self.config = copy.deepcopy(self.config_bak)

            if self.config.training.get("warmup_pointsdist", False):
                if self.current_step < 1000:
                    self.config.training.l2_loss_weight = 0.0
                    self.config.training.perceptual_loss_weight = 0.0
                    self.config.training.pointsdist_loss_weight = 0.1
                    self.config.model.clip_xyz = (
                        False  # turn off xyz clipping for warmup
                    )

            if self.config.training.get("l2_warmup_steps", 0) > 0:
                if self.current_step < self.config.training.l2_warmup_steps:
                    self.config.training.perceptual_loss_weight = 0.0
                    self.config.training.lpips_loss_weight = 0.0
        else:
            self.config = self.config_bak

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        overview = edict(
            pc_tokenizer=count_train_params(self.pc_tokenizer),
            joint_tokenizer=count_train_params(self.joint_tokenizer),
            transformer=count_train_params(self.transformer)
            + count_train_params(self.transformer_input_layernorm),
            joint_mlp=count_train_params(self.joint_mlp),
            joint_fuze_mlp=count_train_params(self.joint_fuse_mlp),
            skinning_mlp=count_train_params(self.skinning_mlp),
            parents_decoder=count_train_params(self.parents_decoder),
            diffloss=count_train_params(self.diffloss),
            start_token=self.start_token.numel(),
        )
        return overview

    def run_layers(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer))):
                if type(concat_nerf_img_tokens) is tuple:
                    input, custom_attn_mask, start_pos = concat_nerf_img_tokens
                    concat_nerf_img_tokens = self.transformer[i](
                        input, attn_mask=custom_attn_mask, start_pos=start_pos
                    )
                else:
                    concat_nerf_img_tokens = self.transformer[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def concat_parent_candidate_features(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Concate current joint's features with its candidate parents' features

        Args:
            joints (torch.Tensor): [b, n_joints, d]

        Returns:
            torch.Tensor: [b, n_joints, n_joints, 2 * d]
        """
        b, n_joints, d = joints.size()
        joints_expand_1 = joints.unsqueeze(1).expand(
            -1, n_joints, -1, -1
        )  # [b, n_joints, n_joints, d]
        joints_expand_2 = joints.unsqueeze(2).expand(
            -1, -1, n_joints, -1
        )  # [b, n_joints, n_joints, d]
        concated_outputs = torch.cat((joints_expand_2, joints_expand_1), dim=-1)

        return concated_outputs

    def concat_pc_joint_features(
        self, pc_tokens: torch.Tensor, joints_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Concate pc_tokens with joints_tokens

        Args:
            pc_tokens (torch.Tensor): [b, n, d]
            joints_tokens (torch.Tensor): [b, n_joints, d]

        Returns:
            torch.Tensor: [b, n_joints, n, 2 * d]
        """
        b, n, d = pc_tokens.size()
        n_joints = joints_tokens.size(1)
        pc_tokens_expand = pc_tokens.unsqueeze(1).expand(
            -1, n_joints, -1, -1
        )  # [b, n_joints, n, d]
        joints_tokens_expand = joints_tokens.unsqueeze(2).expand(
            -1, -1, n, -1
        )  # [b, n_joints, n, d]
        concated_outputs = torch.cat(
            (pc_tokens_expand, joints_tokens_expand), dim=-1
        )  # [b, n_joints, n, 2 * d]
        return concated_outputs

    @torch.no_grad()
    def generate_sequence(
        self, input, create_visual=True, save_skeleton=False, compute_loss=True
    ):
        start_time = time.time()
        generated_joints = []
        generate_joints_tokens = torch.empty(1, 0, self.config.model.transformer.d).to(
            self.device
        )
        generated_joint_tokens_cur = torch.empty(
            1, 0, self.config.model.transformer.d
        ).to(
            self.device
        )  # joint tokens with current index information
        generated_parent_idx = []
        generated_parent_score = []
        checkpoint_every = self.config.training.grad_checkpoint_every

        pointcloud = input["pointcloud"]  # [b, n, 3]
        normals = input["normals"]  # [b, n, 3]
        if "full_pointcloud" in input.keys():
            full_pointcloud = input["full_pointcloud"]
            full_normals = input["full_normals"]
        else:
            full_pointcloud, full_normals = None, None
        scale = input["scale"]  # [b, 1]
        center = input["center"]  # [b, 3]

        b, n, _ = pointcloud.size()
        assert b == 1, "Currently only support batch size 1"
        pointcloud_input = torch.cat((pointcloud, normals), dim=-1)  # [b, n, 6]
        pc_tokens_s = self.pc_tokenizer(pointcloud_input)  # [b, n, d]
        start_token = self.start_token.expand(pc_tokens_s.size(0), 1, -1)  # [b, 1, d]

        input_tokens = torch.cat((pc_tokens_s, start_token), dim=1)  # [b, n + 1, d]

        generated_joints_inv_skinning = torch.empty(1, 0, n).to(self.device)
        start_pos = 0

        total_diff_time = 0
        total_transformer_time = 0

        custom_attn_mask_all = self.get_custom_attn_mask(
            n, 64
        )  # [n + n_joint, n + n_joint]

        # Generate the sequence
        while len(generated_joints) < self.config.model.joints_tokenizer.n_joints:
            input_tokens = self.transformer_input_layernorm(input_tokens)

            # Construct the custom attention mask
            custom_attn_mask = custom_attn_mask_all[
                start_pos : start_pos + input_tokens.size(1),
                : (n + len(generated_joints) + 1),
            ]  # [n + n_joint, n + n_joint]
            transformer_time = time.time()
            for i in range(0, len(self.transformer), checkpoint_every):
                transformer_input = (input_tokens, custom_attn_mask, start_pos)
                input_tokens = torch.utils.checkpoint.checkpoint(
                    self.run_layers(i, i + checkpoint_every),
                    transformer_input,
                    use_reentrant=False,
                )
            transformer_time = time.time() - transformer_time
            total_transformer_time += transformer_time

            if start_pos == 0:
                pc_tokens = input_tokens[:, :n, :]  # [b, n, d]
            start_pos += input_tokens.size(1)
            next_joint_token = input_tokens[:, -1:, :]  # [b, 1, d]

            # Generate joint positions
            diff_time = time.time()
            joints = self.diffloss.sample(next_joint_token.reshape(1, -1))  # [b * 1, 3]
            diff_time = time.time() - diff_time
            total_diff_time += diff_time

            generated_joints.append(
                joints.to(torch.float32).detach().cpu().numpy()[0]
            )  # [n_joints, 3]
            generate_joints_tokens = torch.cat(
                (generate_joints_tokens, next_joint_token), dim=1
            )  # [b, n_joints, d]

            # Parent prediction
            n_joints = len(generated_joints)
            ## Compose joint tokens with current joint's features
            generated_joints_tensor = (
                torch.from_numpy(np.array(generated_joints))
                .to(next_joint_token.device)
                .view(b, n_joints, -1)
            )  # [b, n_joints, 3]
            cur_joint_pos_token = self.joint_tokenizer(
                generated_joints_tensor
            )  # [b, n_joints, d]
            cur_joint_index_pos_embedding = (
                self.joint_index_pos_embedding.unsqueeze(0)
                .expand(b, -1, -1)
                .to(next_joint_token.device)[:, :n_joints, :]
            )  # [b, n_joints, d]
            joints_tokens_cur = torch.cat(
                (
                    generate_joints_tokens,
                    cur_joint_index_pos_embedding,
                    cur_joint_pos_token,
                ),
                dim=-1,
            )  # [b, 1, 3 * d]
            joints_tokens_cur = self.joint_fuse_mlp(joints_tokens_cur)  # [b, 1, d]
            generated_joint_tokens_cur = torch.cat(
                (generated_joint_tokens_cur, joints_tokens_cur[:, -1:, :]), dim=1
            )  # [b, n_joints, d]
            joint_parent_candidates = self.concat_parent_candidate_features(
                joints_tokens_cur
            )  # [b, n_joints, n_joints, 2 * d]
            parents = self.parents_decoder(joint_parent_candidates).view(
                b, n_joints, n_joints
            )  # [b, n_joints, n_joints]
            # Pad to default shape of [max_n_joints, max_n_joints] with zeros
            parents_score = (
                torch.ones(
                    b,
                    self.config.model.joints_tokenizer.n_joints,
                    self.config.model.joints_tokenizer.n_joints,
                ).to(self.device)
                * -10000
            )
            parents_score[:, :n_joints, :n_joints] = parents
            # Update the parent score of newly generated joint
            generated_parent_score.append(
                parents_score.to(torch.float32)
                .detach()
                .cpu()
                .numpy()[0, (n_joints - 1) : n_joints]
            )
            parents_pred = F.softmax(
                parents_score[:, (n_joints - 1) : n_joints], dim=-1
            ).to(
                torch.float32
            )  # [b, 1, n_joints]
            parents_pred_idx = torch.argmax(parents_pred, dim=-1)  # [b, 1]
            generated_parent_idx.append(parents_pred_idx.detach().cpu().numpy()[-1:])

            # Stop condition
            if (
                len(generated_joints) - 1 == parents_pred_idx[-1]
                and len(generated_joints) > 1
            ):
                n_joints -= 1
                generate_joints_tokens = generate_joints_tokens[:, :-1, :]
                generated_joint_tokens_cur = generated_joint_tokens_cur[:, :-1, :]
                generated_parent_score = generated_parent_score[:-1]
                generated_joints = generated_joints[:-1]
                generated_joints_inv_skinning = generated_joints_inv_skinning[:, :-1, :]
                generated_parent_idx = generated_parent_idx[:-1]
                break

            # Construct new input tokens
            new_joints_input = (
                torch.from_numpy(np.array(generated_joints))
                .to(next_joint_token.device)
                .view(1, n_joints, -1)
            )  # [b, n_joints, 3]
            new_parent_idx = (
                torch.from_numpy(np.array(generated_parent_idx))
                .to(next_joint_token.device)
                .view(1, n_joints)
            )  # [b, n_joints]
            new_joints_pos_tokens = self.joint_tokenizer(
                new_joints_input
            )  # [b, n_joints, d]
            new_joints_parent_tokens = torch.gather(
                new_joints_pos_tokens,
                1,
                new_parent_idx.unsqueeze(-1).expand(
                    -1, -1, new_joints_pos_tokens.size(-1)
                ),
            )  # [b, n_joints, d]
            new_joints_index_pos_embedding = (
                self.joint_index_pos_embedding.unsqueeze(0)
                .expand(b, -1, -1)
                .to(next_joint_token.device)
            )  # [b, n_joints, d]
            new_joints_index_pos_embedding = new_joints_index_pos_embedding[
                :, :n_joints, :
            ]  # [b, n_joints, d]
            new_joints_parent_index_pos_embedding = torch.gather(
                new_joints_index_pos_embedding,
                1,
                new_parent_idx.unsqueeze(-1).expand(
                    -1, -1, new_joints_index_pos_embedding.size(-1)
                ),
            )  # [b, n_joints, d]
            new_joints_tokens = torch.cat(
                (
                    new_joints_index_pos_embedding,
                    new_joints_pos_tokens,
                    new_joints_parent_index_pos_embedding,
                    new_joints_parent_tokens,
                ),
                dim=-1,
            )  # [b, n_joints, 3 * d]
            new_joints_tokens = self.joint_mlp(new_joints_tokens)  # [b, n_joints, d]
            input_tokens = new_joints_tokens[:, -1:, :]  # [b, n_joints, d]

        # compute time
        elapsed_time = time.time() - start_time
        print(f"Total transformer time: {total_transformer_time:.2f}s")
        print(f"Total diff time: {total_diff_time:.2f}s")
        print(f"Time elapsed: {elapsed_time:.2f}s")
        ## Compose skinning token
        n_joints = len(generated_joints)
        if full_pointcloud is not None:
            # Iterate through all points
            points_batch = torch.split(full_pointcloud, n, dim=1)
            normals_batch = torch.split(full_normals, n, dim=1)
            full_pc_tokens = []
            generated_joints_inv_skinning = torch.empty(
                1, generated_joint_tokens_cur.shape[1], 0
            ).to(self.device)
            custom_attn_mask_cur = self.get_custom_attn_mask(n, 1)
            for points_cur, normals_cur in zip(points_batch, normals_batch):
                pointcloud_input_cur = torch.cat((points_cur, normals_cur), dim=-1)
                pc_tokens_s_cur = self.pc_tokenizer(pointcloud_input_cur)  # [b, n, d]
                start_token = self.start_token.expand(pc_tokens_s_cur.size(0), 1, -1)
                input_tokens_cur = torch.cat((pc_tokens_s_cur, start_token), dim=1)

                input_tokens_cur = self.transformer_input_layernorm(input_tokens_cur)
                custom_attn_mask_cur = self.get_custom_attn_mask(
                    pointcloud_input_cur.shape[1], 1
                )
                for i in range(0, len(self.transformer), checkpoint_every):
                    transformer_input = (input_tokens_cur, custom_attn_mask_cur, 0)
                    input_tokens_cur = torch.utils.checkpoint.checkpoint(
                        self.run_layers(i, i + checkpoint_every),
                        transformer_input,
                        use_reentrant=False,
                    )
                pc_tokens_cur = input_tokens_cur[
                    :, : pointcloud_input_cur.shape[1], :
                ]  # [b, n, d]
                skinning_tokens = self.concat_pc_joint_features(
                    pc_tokens_cur, generated_joint_tokens_cur
                )  # [b, n_joints, n, 2 * d]
                skinning_pred_cur = self.skinning_mlp(skinning_tokens).view(
                    b, n_joints, -1
                )  # [b, n_joints, n]
                generated_joints_inv_skinning = torch.cat(
                    (generated_joints_inv_skinning, skinning_pred_cur), dim=2
                )  # [b, n_joints, n]

        else:
            skinning_tokens = self.concat_pc_joint_features(
                pc_tokens, generated_joint_tokens_cur
            )  # [b, n_joints, n, 2 * d]
            skinning_pred = self.skinning_mlp(skinning_tokens).view(
                b, n_joints, -1
            )  # [b, n_joints, n]
            generated_joints_inv_skinning = torch.cat(
                (generated_joints_inv_skinning, skinning_pred), dim=1
            )  # [b, n_joints, n]

        # Construct the npz file
        # Scale back joint position first
        joints_npz = np.array(generated_joints)
        joints_npz = (
            joints_npz * scale.detach().cpu().numpy() + center.detach().cpu().numpy()
        )
        parents_npz = np.array(generated_parent_idx).reshape(-1)
        joints_inv_skinning_npz = (
            generated_joints_inv_skinning.to(torch.float32).detach().cpu().numpy()[0].T
        )  # [n_joints, n]
        if full_pointcloud is not None:
            pointcloud_npz = (
                full_pointcloud.to(torch.float32).detach().cpu().numpy()[0]
                * scale.detach().cpu().numpy()
                + center.detach().cpu().numpy()
            )
        else:
            pointcloud_npz = (
                input["pointcloud"].to(torch.float32).detach().cpu().numpy()[0]
                * scale.detach().cpu().numpy()
                + center.detach().cpu().numpy()
            )
        data_npz_dict = {
            "joints": joints_npz,
            "parents": parents_npz,
            "skinning_weights": joints_inv_skinning_npz,
            "pointcloud": pointcloud_npz,
        }

        result = edict(
            input=input,
            npz_dict=data_npz_dict,
            elapsed_time=elapsed_time,
        )
        return result

    @torch.no_grad()
    def save_results(
        self, out_dir, result, batch, steps, save_all=False, save_skeleton=False
    ):
        os.makedirs(out_dir, exist_ok=True)
        item_idx = batch["item_idx"][0]
        # save the npz file
        if result.npz_dict is not None:
            npz_path = os.path.join(out_dir, f"{item_idx}.npz")
            np.savez(npz_path, **result.npz_dict)
