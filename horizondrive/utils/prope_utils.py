from functools import partial
from typing import Callable, Optional, Tuple

import torch


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, seqlen, num_heads, feat_dim)
    projmat: torch.Tensor,  # (batch, cameras, 4, 4)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = projmat.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    assert projmat.shape == (batch, cameras, 4, 4)
    output = torch.einsum(
        "bcij,bcpnkj->bcpnki",
        projmat.to(feats.device),
        feats.reshape((batch, cameras, -1, num_heads, feat_dim // 4, 4)),
    ).reshape(feats.shape)

    return output


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000.0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # shape: (S, D/2)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def _rope_apply(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    freqs: torch.Tensor,  # (max_seqlen, feat_dim // 2)
    seq_size: torch.Tensor,
    emd_dim: int = 0,
) -> torch.Tensor:
    """Apply RoPE coefficients to features."""
    # b, n, s, d = feats.shape
    b, s, n, d = feats.shape
    f, h, w = seq_size[0].item(), seq_size[1].item(), seq_size[2].item()
    freq_len = seq_size[emd_dim].item()
    
    if emd_dim == 0:
        view_dim = (f, 1, 1, -1)
    elif emd_dim == 1:
        view_dim = (1, h, 1, -1)
    elif emd_dim == 2:
        view_dim = (1, 1, w, -1)
    else:
        raise ValueError(f"Invalid emd_dim: {emd_dim}. Must be 0, 1, or 2.")
    freqs_i = freqs[:freq_len, :].view(*view_dim).expand(f, h, w, -1).reshape(s, 1, -1).to(feats.device)
    output = []
    for i in range(b):
        x_i = torch.view_as_complex(feats[i].to(torch.float32).reshape(s, n, -1, 2))
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)  # shape: (n, s, d)
        output.append(x_i)
    
    return torch.stack(output).float().to(feats.device)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_old(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # # split freqs
    # freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
class PropeDotProductAttention(torch.nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        freq_base: float = 10000.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.freq_base = freq_base
        self.freq_scale = freq_scale

        frame_dim = head_dim - 4 * (head_dim // 6)
        height_dim = 2 * (head_dim // 6)
        width_dim = 2 * (head_dim // 6)

        self.frame_proj_dim = ((frame_dim // 2) // 4) * 4
        self.height_proj_dim = ((height_dim // 2) // 4) * 4
        self.width_proj_dim = ((width_dim // 2) // 4) * 4

        self.frame_rope_dim = frame_dim - self.frame_proj_dim
        self.height_rope_dim = height_dim - self.height_proj_dim
        self.width_rope_dim = width_dim - self.width_proj_dim

        theta = self.freq_base * self.freq_scale
        self.freqs =[
            rope_params(max_seq_len, frame_dim, theta),
            rope_params(max_seq_len, height_dim, theta),
            rope_params(max_seq_len, width_dim, theta)
        ]
        self.prope_weight = 1.0

    def set_projection_matrices(self, proj_mats=None):
        if proj_mats is None:
            raise ValueError("Projection matrices must be provided.")
        self.P = proj_mats
        self.P_inv = torch.inverse(self.P)
        self.P_T = proj_mats.transpose(-1, -2)
    
    def construct_transforms(self, seq_size, projmat):
        """Construct transforms for RoPE application."""
        return [
            (partial(_rope_apply, freqs=self.freqs[0][:, :(self.frame_rope_dim // 2)], seq_size=seq_size, emd_dim=0), self.frame_rope_dim),
            (partial(_apply_tiled_projmat, projmat=projmat), self.frame_proj_dim),
            (partial(_rope_apply, freqs=self.freqs[1][:, :(self.height_rope_dim // 2)], seq_size=seq_size, emd_dim=1), self.height_rope_dim),
            (partial(_apply_tiled_projmat, projmat=projmat), self.height_proj_dim),
            (partial(_rope_apply, freqs=self.freqs[2][:, :(self.width_rope_dim // 2)], seq_size=seq_size, emd_dim=2), self.width_rope_dim),
            (partial(_apply_tiled_projmat, projmat=projmat), self.width_proj_dim),
        ]
    
    def set_prope_weight(self, prope_weight: float):
        """Set the weight for ProPE."""
        assert 0 <= prope_weight <= 1, "prope_weight must be in [0, 1]."
        self.prope_weight = prope_weight

    def forward(self, q, k, grid_sizes, **kwargs):
        d = self.head_dim
        # Block-diagonal transforms to the inputs and outputs of the attention operator.
        assert d % 4 == 0

        # 保留前半部分
        transforms_q = self.construct_transforms(grid_sizes[0], self.P_T)
        transforms_kv = self.construct_transforms(grid_sizes[0], self.P_inv)
        q_new = _apply_block_diagonal(q, transforms_q)
        k_new = _apply_block_diagonal(k, transforms_kv)

        if self.prope_weight < 1.0:
            grid_sizes = grid_sizes.to(q.device)
            q_old = rope_apply_old(q, grid_sizes, self.freqs)
            k_old = rope_apply_old(k, grid_sizes, self.freqs)

            q = self.prope_weight * q_new + (1 - self.prope_weight) * q_old
            k = self.prope_weight * k_new + (1 - self.prope_weight) * k_old
        else:
            q = q_new
            k = k_new

        return q, k

    def device(self) -> torch.device:
        """Override to ensure buffers are moved correctly."""
        assert hasattr(self, 'freqs'), "Coefficients not initialized."
        return self.freqs[0].device
    
    def to(self, device: Optional[torch.device] = None):
        """Override to ensure buffers are moved correctly."""
        assert hasattr(self, 'freqs'), "Coefficients not initialized."
        self.freqs = [
            f.to(device=device) if isinstance(f, torch.Tensor) else f for f in self.freqs
        ]


@torch.amp.autocast('cuda', enabled=False)
def get_prope_fn(
    head_dim: int,
    projmat: torch.Tensor,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor],
           Callable[[torch.Tensor], torch.Tensor],
           Callable[[torch.Tensor], torch.Tensor]]:
    """Get ProPE functions for q, kv, and o."""
    transforms_q = [
        (partial(_apply_tiled_projmat, projmat=projmat.transpose(-1, -2)), head_dim),
    ]
    proj_mats_inv = torch.inverse(projmat.to(torch.float32)).to(projmat.dtype)
    transforms_kv = [
        (partial(_apply_tiled_projmat, projmat=proj_mats_inv), head_dim),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, projmat=projmat), head_dim),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


if __name__ == "__main__":
    def test_rope_old(q, k, grid_sizes, device):
        b, s, n, d = q.shape
        freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ]
        freqs = [f.to(device) for f in freqs]
        grid_sizes = grid_sizes.to(device)
        q = rope_apply_old(q, grid_sizes, freqs)
        k = rope_apply_old(k, grid_sizes, freqs)

        return q, k
    

    f = 7
    h = 30
    w = 30
    b = 1
    n = 8
    d = 128

    device = torch.device("cuda:0")
    q = torch.randn(b, f * h * w, n,  d).to(device)
    k = torch.randn(b, f * h * w, n, d).to(device)
    v = torch.randn(b, f * h * w, n, d).to(device)

    proj_mats = torch.randn(b, f, 4, 4)
    proj_inv_mats = torch.inverse(proj_mats)


    grid_sizes = torch.tensor([[f, h, w]], device=device)  # (batch, 3)
    attn = PropeDotProductAttention(d, max_seq_len=1024)
    attn.set_projection_matrices(proj_mats, proj_inv_mats)
    q_new, k_new = attn(q, k, grid_sizes, prope_weight=1.0)

    q_old, k_old = test_rope_old(q, k, grid_sizes= grid_sizes, device=device)
    print(f"q_new shape: {q_new.shape}, k_new shape: {k_new.shape}")
    print(f"q_old shape: {q_old.shape}, k_old shape: {k_old.shape}")


    frame_dim = d - 4 * (d // 6)
    height_dim = 2 * (d // 6)
    width_dim = 2 * (d // 6)

    frame_proj_dim = ((frame_dim // 2) // 4) * 4
    height_proj_dim = ((height_dim // 2) // 4) * 4
    width_proj_dim = ((width_dim // 2) // 4) * 4


    # 验证q和q_old在frame_rope_dim, height_rope_dim，width_rope_dim上的值是否相同
    assert torch.allclose(q_new[..., :frame_proj_dim], q_old[..., :frame_proj_dim], atol=1e-6), "Frame RoPE values do not match."
    assert torch.allclose(q_new[..., frame_dim: frame_dim + height_proj_dim], q_old[..., frame_dim: frame_dim + height_proj_dim], atol=1e-6), "Height RoPE values do not match."
    assert torch.allclose(q_new[..., frame_dim + height_dim : frame_dim + height_dim + width_proj_dim], q_old[..., frame_dim + height_dim : frame_dim + height_dim + width_proj_dim], atol=1e-6), "Width RoPE values do not match."
