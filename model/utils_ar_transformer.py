import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from .utils_transformer import MLP

# try:
#     import xformers.ops as xops
# except ImportError as e:
#     print("Please install xformers to use flashatt v2")
#     raise e

# https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L162-L168

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  
  
class MLP(nn.Module):
    """
    MLP layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    Ignore: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L62
    """

    def __init__(
        self,
        d,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        mlp_dim=None,
    ):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = d * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d, bias=mlp_bias),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class CustomCausalAttention(nn.Module):
    """
    Custom causal attention module, only do causal attention within joints-joints, self-attention within pointclouds
    """

    def __init__(
        self,
        d,
        d_head,
        n_pc,
        n_joints,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        use_flashatt_v2=False,
        device="cuda",
    ):
        super(CustomCausalAttention, self).__init__()
        assert (
            d % d_head == 0
        ), f"Token dimension {d} should be divisible by head dimension {d_head}"
        self.d = d
        self.d_head = d_head
        self.n_pc = n_pc
        self.n_joints = n_joints
        self.attn_dropout = attn_dropout

        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

        self.use_flashatt_v2 = use_flashatt_v2

        # Construct the custom attention mask
        #   P P P P | J J J
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # P 1 1 1 1 | 0 0 0
        # - - - - - | - - -
        # J 1 1 1 1 | 1 0 0
        # J 1 1 1 1 | 1 1 0
        # J 1 1 1 1 | 1 1 1
        self.attention_mask_flash = torch.ones(
            n_pc + n_joints, n_pc + n_joints
        ) * float("-inf")
        self.attention_mask_flash[:n_pc, :n_pc] = 0
        self.attention_mask_flash[n_pc:, :n_pc] = 0
        tri_mask = torch.tril(torch.ones(n_joints, n_joints))
        tri_mask = torch.masked_fill(tri_mask, tri_mask == 0.0, float("-inf"))
        tri_mask = torch.masked_fill(tri_mask, tri_mask == 1.0, 0)
        self.attention_mask_flash[n_pc:, n_pc:] = tri_mask
        self.attention_mask = ~self.attention_mask_flash.to(torch.bool).to(device)

        # kv cache
        self.max_batch_size = 1
        self.max_seq_len = 1024 + 128  # TODO 1024 is the point cloud token size
        self.register_buffer(
            "k_cache",
            torch.zeros(
                self.max_batch_size,
                self.d // self.d_head,
                self.max_seq_len,
                self.d_head,
            ),
            persistent=False,
        )
        self.k_cache.to(torch.float16).to(device)
        self.register_buffer(
            "v_cache",
            torch.zeros(
                self.max_batch_size,
                self.d // self.d_head,
                self.max_seq_len,
                self.d_head,
            ),
            persistent=False,
        )
        self.v_cache.to(torch.float16).to(device)

    def forward(self, x, subset_attention_size=None, attn_mask=None, start_pos=0):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d] / [b, l, d]
            subset_attention_size: The size of the subset to perform attention on
            attn_mask: The attention mask to use
        """
        batch_size, seq_len, _ = x.shape

        if attn_mask is None:
            assert x.shape[-2] == (
                self.n_pc + self.n_joints
            ), f"Input tensor shape {x.shape} does not match the expected shape"
            attn_mask = self.attention_mask
        else:
            attn_mask = attn_mask.to(torch.bool).to(x.device)

        # token split, multi-head attention, token cat
        q, k, v = self.to_qkv(x).split(self.d, dim=2)  # [batch_size, seq_len, d] x 3

        if self.use_flashatt_v2:
            raise NotImplementedError("Flash attention v2 is not supported yet")
        else:
            # Rearrange the input tensor to (batch, heads, seq_len, dim)
            q, k, v = (
                rearrange(q, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(k, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(v, "b l (nh dh) -> b nh l dh", dh=self.d_head),
            )

            # Update KV cache
            if start_pos < self.max_seq_len:
                end_pos = min(start_pos + seq_len, self.max_seq_len)
                self.k_cache[:, :, start_pos:end_pos, :] = k[
                    :, :, : end_pos - start_pos, :
                ]
                self.v_cache[:, :, start_pos:end_pos, :] = v[
                    :, :, : end_pos - start_pos, :
                ]

            # https://discuss.pytorch.org/t/flash-attention/174955/14
            dropout_p = self.attn_dropout if self.training else 0.0
            if subset_attention_size is not None and subset_attention_size < q.shape[2]:
                raise NotImplementedError("Subset attention is not supported yet")
            else:
                # Use the cached kv
                k_to_use = self.k_cache[:batch_size, :, :end_pos, :].to(
                    torch.float16
                )  # bnld
                v_to_use = self.v_cache[:batch_size, :, :end_pos, :].to(torch.float16)

                if attn_mask is not None:
                    cache_attn_mask = attn_mask[:, : start_pos + seq_len]

                x = F.scaled_dot_product_attention(
                    q,
                    k_to_use,
                    v_to_use,
                    dropout_p=dropout_p,
                    attn_mask=cache_attn_mask,
                )
                x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.attn_fc_dropout(self.fc(x))
        return x

class CustomTransformerBlock(nn.Module):
    """
    Custom Transformer block
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    Note: move drop_path to SelfAttention and MLP
    """
    def __init__(
        self,
        d,
        d_head,
        n_pc,
        n_joints,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
    ):
        super(CustomTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.attn = CustomCausalAttention(
            d,
            d_head,
            n_pc,
            n_joints,
            attn_qkv_bias,
            attn_dropout,
            attn_fc_bias,
            attn_fc_dropout,
        )
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        self.mlp = MLP(d, mlp_ratio, mlp_bias, mlp_dropout)

    def forward(self, x, subset_attention_size=None, attn_mask=None, start_pos=0):
        x = x + self.attn(
            self.norm1(x),
            subset_attention_size=subset_attention_size,
            attn_mask=attn_mask,
            start_pos=start_pos,
        )
        x = x + self.mlp(self.norm2(x))
        return x
