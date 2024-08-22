""""
Caution: The image tokenizer layers are from Minicpm, waiting for integration
"""


import os
import re
import torch
import argparse
import json
import numpy as np
import time

from gguf import *
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipVisionConfig

TEXT = "clip.text"
VISION = "clip.vision"



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h_size, grid_w_size = grid_size, grid_size
    else:
        grid_h_size, grid_w_size = grid_size[0], grid_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def _replace_name_resampler(s, v):
    if re.match("resampler.pos_embed", s):
        return {
            s: v,
            re.sub("pos_embed", "pos_embed_k", s): torch.from_numpy(get_2d_sincos_pos_embed(4096, (70, 70))),
        }
    if re.match("resampler.proj", s):
        return {
            re.sub("proj", "pos_embed_k", s): torch.from_numpy(get_2d_sincos_pos_embed(4096, (70, 70))),
            re.sub("proj", "proj.weight", s): v.transpose(-1, -2).contiguous(),
        }
    if re.match("resampler.attn.in_proj_.*", s):
        return {
            re.sub("attn.in_proj_", "attn.q.", s): v.chunk(3, dim=0)[0],
            re.sub("attn.in_proj_", "attn.k.", s): v.chunk(3, dim=0)[1],
            re.sub("attn.in_proj_", "attn.v.", s): v.chunk(3, dim=0)[2],
        }
    return {s: v}

def _replace_name_vit(s,v):
    s = "vision_model." + s
    if re.match("vision_model.embeddings.position_embedding", s):
        v = v.unsqueeze(0)
        return {s: v}
    return {s: v}

def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

def should_skip_tensor(name: str, has_text: bool, has_vision: bool, has_minicpmv: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    if has_minicpmv and name in ["visual_projection.weight"]:
        return True

    if name.startswith("v") and not has_vision:
        return True

    if name.startswith("t") and not has_text:
        return True

    return False


def get_tensor_name(name: str) -> str:
    if "projection" in name:
        return name
    if "mm_projector" in name:
        name = name.replace("model.mm_projector", "mm")
        name = re.sub(r'mm\.mlp\.mlp', 'mm.model.mlp', name, count=1)
        name = re.sub(r'mm\.peg\.peg', 'mm.model.peg', name, count=1)
        return name

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class print_time():
    def __init__(self, task):
        self.task = task
        
    def __enter__(self):
        print(f"🟡 {self.task}")
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'🟢 time used: [{time.time() - self.t:.03f}] secs')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surgery_dir", type=str)
    parser.add_argument('--version', type=str, default='siglip_kosmos_phi3_4k_instruct', help='help identify the version of the saved ckpt')
    parser.add_argument("--use_f32", action="store_true", help="Use f32 instead of f16")
    parser.add_argument("--text-only", action="store_true", required=False,
                help="Save a text-only model. It can't be used to encode images")
    parser.add_argument("--vision-only", action="store_true", required=False,
                help="Save a vision-only model. It can't be used to encode texts")
    parser.add_argument("--xgenmm_projector", help="Path to minicpmv.projector file. If specified, save an image encoder for XgenMM models.")
    parser.add_argument("--xgenmm_vit", help="Path to vit file. ")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.text_only and args.vision_only:
        raise ValueError("--text-only and --image-only arguments cannot be specified at the same time.")

    if args.use_f32:
        print("WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")
    
    ckpt_dir = f"{args.surgery_dir}/{args.version}"
    if args.xgenmm_projector is None:
        args.xgenmm_projector = f"{ckpt_dir}/xgenmm.projector"
    if args.xgenmm_vit is None:
        args.xgenmm_vit = f"{ckpt_dir}/vision_encoder/xgenmm.vision_encoder"
    output_dir = f"{ckpt_dir}/gguf"

    ftype_str = ["f32", "f16"]

    ftype = 1
    if args.use_f32:
        ftype = 0        
    
    with print_time("Loading vision encoder"):
        vision_encoder_config_path = f"{args.surgery_dir}/{args.version}/vision_encoder/config.json"
        with open(vision_encoder_config_path, 'r') as f:
            vision_config = json.load(f)
        vision_encoder_config = SiglipVisionConfig(**vision_config)
        # we don't have this config: "initializer_range": 0.02,
        # print(vision_encoder_config)

        vision_encoder = SiglipVisionTransformer(vision_encoder_config)
        vision_encoder_ckpt = torch.load(f'{ckpt_dir}/vision_encoder/xgenmm.vision_encoder')
        vision_encoder.load_state_dict(vision_encoder_ckpt)

    fname_middle = None
    has_text_encoder = True
    has_vision_encoder = True
    has_minicpmv_projector = False
    if args.text_only:
        fname_middle = "text-"
        has_vision_encoder = False
    elif args.xgenmm_projector is not None:
        fname_middle = "mmproj-"
        has_text_encoder = False
        has_minicpmv_projector = True
    elif args.vision_only:
        fname_middle = "vision-"
        has_text_encoder = False
    else:
        fname_middle = ""

    # output_dir = f"{ckpt_dir}/gguf"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_prefix = os.path.basename(output_dir).replace("ggml_", "")  # seems not been used
    fname_out = os.path.join(output_dir, f"{fname_middle}model-{ftype_str[ftype]}.gguf")

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_bool("clip.has_text_encoder", has_text_encoder)
    fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
    fout.add_bool("clip.has_minicpmv_projector", has_minicpmv_projector)
    fout.add_file_type(ftype)
    
    if args.text_only:
        fout.add_description("text-only CLIP model")
    elif args.vision_only and not has_minicpmv_projector:
        fout.add_description("vision-only CLIP model")
    elif has_minicpmv_projector:
        # fout.add_description("image encoder for XgenMM model")
        # # add projector type
        # fout.add_string("clip.projector_type", "PerceiverResampler")
        # change to real projector later
        fout.add_description("image encoder for MiniCPM-V")
        # add projector type
        fout.add_string("clip.projector_type", "resampler")
    else:
        fout.add_description("two-tower CLIP model")

    if has_vision_encoder:
        """
        In siglip config, we have following keys
            used: "image_size", "patch_size", "hidden_size", "intermediate_size"
                        "num_attention_heads", "layer_norm_eps", "num_hidden_layers", "hidden_act"
            unused: "attention_dropout", "model_type", "num_channels"
        """
        fout.add_uint32("clip.vision.image_size", vision_config["image_size"])
        fout.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
        fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vision_config["hidden_size"])
        fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), vision_config["intermediate_size"])
        # TODO: need to check the value of projection_dim; follow minicpmv to set it as 0
        fout.add_uint32("clip.vision.projection_dim", 0)
        fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vision_config["num_attention_heads"])
        fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), vision_config["layer_norm_eps"])
        # TODO: chekck this as it might causes bugs
        # orginial llaval implementation:
        # block_count = vision_config["num_hidden_layers"] - 1 if has_xgenmm_projector else vision_config["num_hidden_layers"]
        # we are different from llama1.6, which used the second to the last layer's hidden states as the image features.
        block_count = vision_config["num_hidden_layers"] 
        fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)
        print(KEY_BLOCK_COUNT)
        # xgenmm use anyres with grids configuration
        # 1*2, 2*1, 2*2, 3*1, 1*3, the same as the llava1.6, we just hard code it here
        image_grid_pinpoints = [336, 672, 672, 336, 672, 672, 1008, 336, 336, 1008]
        fout.add_array("clip.vision.image_grid_pinpoints", image_grid_pinpoints)
        
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        fout.add_array("clip.vision.image_mean", image_mean)
        fout.add_array("clip.vision.image_std", image_std)
        
        # TODO: need to check; vision_config["hidden_act"] is gelu_pytorch_tanh
        use_gelu = "gelu" in vision_config["hidden_act"].lower()
        fout.add_bool("clip.use_gelu", use_gelu)
        
    # need to replace this part to PerceiverResampler
    if has_minicpmv_projector:
        projector = torch.load(args.xgenmm_projector)
        new_state_dict = {}
        for k_, v_ in projector.items():
            kvs = _replace_name_resampler(k_, v_)
            for nk, nv in kvs.items():
                new_state_dict[nk] = nv
        projector = new_state_dict
        for name, data in projector.items():
            name = get_tensor_name(name)
            data = data.squeeze().numpy()

            n_dims = len(data.shape)

            if ftype == 1:
                if name[-7:] == ".weight" and n_dims == 2:
                    print("  Converting to float16")
                    data = data.astype(np.float16)
                    ftype_cur = 1
                else:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype_cur = 0
            else:
                if data.dtype != np.float32:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    # ftype_cur = 0
                ftype_cur = 0

            fout.add_tensor(name, data)
            print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")

        print("Projector tensors added\n")
    # end


    # for VIT model
    state_dict = vision_encoder.state_dict()
    new_state_dict = {}
    for k_, v_ in state_dict.items():
        kvs = _replace_name_vit(k_, v_)
        for nk, nv in kvs.items():
            # split in_proj_weight to q_proj_weight, k_proj_weight, v_proj_weight
            if nk == "vision_model.head.attention.in_proj_weight":
                dim = int(nv.shape[0] / 3)
                nk_1 = "vision_model.head.attention.q_proj_weight"
                nv_1 = nv[:dim, :]
                nk_2 = "vision_model.head.attention.k_proj_weight"
                nv_2 = nv[dim:2*dim, :]
                nk_3 = "vision_model.head.attention.v_proj_weight"
                nv_3 = nv[2*dim:, :]
                new_state_dict[nk_1] = nv_1
                new_state_dict[nk_2] = nv_2
                new_state_dict[nk_3] = nv_3
            # split in_proj_bias to q_proj_bias, k_proj_bias, v_proj_bias
            elif nk == "vision_model.head.attention.in_proj_bias":
                dim = int(nv.shape[0] / 3)
                nk_1 = "vision_model.head.attention.q_proj_bias"
                nv_1 = nv[:dim]
                nk_2 = "vision_model.head.attention.k_proj_bias"
                nv_2 = nv[dim:2*dim]
                nk_3 = "vision_model.head.attention.v_proj_bias"
                nv_3 = nv[2*dim:]
                new_state_dict[nk_1] = nv_1
                new_state_dict[nk_2] = nv_2
                new_state_dict[nk_3] = nv_3
            else:
                new_state_dict[nk] = nv


    state_dict = new_state_dict
    for name, data in state_dict.items():
        if should_skip_tensor(name, has_text_encoder, has_vision_encoder, has_minicpmv_projector):
            # we don't need this
            print(f"skipping parameter: {name}")
            continue

        name = get_tensor_name(name)
        data = data.squeeze().numpy()
        
        n_dims = len(data.shape)

        ftype_cur = 0
        if n_dims == 4:
            print(f"tensor {name} is always saved in f16")
            data = data.astype(np.float16)
            ftype_cur = 1
        elif ftype == 1:
            if name[-7:] == ".weight" and n_dims == 2:
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype_cur = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0
        else:
            if data.dtype != np.float32:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

        print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
        fout.add_tensor(name, data)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("Done. Output file: " + fname_out)