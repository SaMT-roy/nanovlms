from llm_lora import *
import timm
import torch
from tokenizers import Tokenizer
from torchvision.transforms import v2 as T
import torchvision.io as io
import json
import pandas as pd
import numpy as np
from typing import List
import os
import cv2
import matplotlib.pyplot as plt
import ast
import re
from dinov3_meta import DinoVisionTransformer

device = 'cpu' # torch.device("cuda:0")
tokenizer = Tokenizer.from_file("/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3/model_weights/tokenizer.json")

pad_id = tokenizer.encode("<empty_output>").ids
bos_id = tokenizer.encode("<|im_start|>").ids
eos_id = tokenizer.encode("<|im_end|>").ids
sep_id = tokenizer.encode("<file_sep>").ids
img_token_id = tokenizer.encode("<filename>").ids

system_prompt1 = tokenizer.encode(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
).ids

system_prompt2 = tokenizer.encode(
    "<|im_end|>\n<|im_start|>system\nAnalyze the provided information and answer the following question.<|im_end|>\n<|im_start|>user\n"
).ids

# Encode the assistant prompt and separator
assistant_prompt = tokenizer.encode(
    "<|im_end|>\n<|im_start|>assistant\n"
).ids

vocab = tokenizer.get_vocab()           
id_to_token = {idx: tok for tok, idx in vocab.items()}

class LMConfig:
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type = 'smollm2-360m'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# Use it in VLM
cfg = LMConfig()

# https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py
def vit_small(**kwargs):
    model = DinoVisionTransformer(
        img_size=256,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        compact_arch_name="vits",
        **kwargs,
    )
    return model

class PixelUnshuffle(torch.nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        """
        Manual implementation of Pixel Unshuffle.
        Input shape: (Batch, Channels, Height, Width) -> (B, C, H, W)
        Output shape: (Batch, Channels * r^2, H/r, W/r)
        """
        b, c, h, w = x.shape
        r = self.downscale_factor
        
        # 1. Split H and W into (H/r, r) and (W/r, r)
        assert h % r == 0 and w % r == 0, \
            f"H and W must be divisible by r={r}, got {h}, {w}"
        
        out_h, out_w = h // r, w // r
        
        # Reshape to (B, C, H/r, r, W/r, r)
        x = x.view(b, c, out_h, r, out_w, r)
        
        # 2. Permute to bring the 'r' dimensions next to the channels
        # Current: (B, C, H/r, r, W/r, r) -> Target: (B, C, r, r, H/r, W/r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        
        # 3. Flatten the (C, r, r) into a single dimension
        x = x.view(b, c * (r * r), out_h, out_w)
        
        return x
    
IMG_SEQ_LEN = 64

class VLM(torch.nn.Module):
    def __init__(self, cfg, activate_lora=False):
        super().__init__()
        # --- Language Model ---
        self.decoder = LanguageModel.from_pretrained(cfg, activate_lora=activate_lora)
        self.d_model = cfg.lm_hidden_dim
        self.img_token_id = img_token_id[0]

        # --- Image Embedding Model (ViT) ---
        # Using a small ViT model as an example
        self.img_emb_model = vit_small()

        self.shuffle_factor = 3
        self.pixel_unshuffle = PixelUnshuffle(self.shuffle_factor)
        self.img_feature_dim = 384*(self.shuffle_factor)**2

        # --- Learnable Query Tokens and Cross-Attention ---
        self.img_seq_len = IMG_SEQ_LEN

        # --- Dense Projector ---
        # This now projects the ViT patch embeddings to the LLM's hidden dimension
        self.dense = torch.nn.Sequential(
                    torch.nn.Linear(self.img_feature_dim, self.d_model),  # expand
                    torch.nn.GELU(),                                      # non-linear activation
                    torch.nn.Linear(self.d_model, self.d_model)           # project down to LM dim
                    )

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=30, temp=1.0):
        img_tensors, tokens = inputs
        img_tensors = img_tensors.to(tokens.device)

        # --- Get ViT Features ---
        vit_tokens = self.img_emb_model.forward_features(img_tensors)['x_norm_patchtokens']

        B, N, D = vit_tokens.shape
        H = W = int(N ** 0.5)

        vit_feat_map = vit_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        vit_feat_map = self.pixel_unshuffle(vit_feat_map)  # (B, 9*D, 8, 8)

        vit_features = vit_feat_map.flatten(2).transpose(1, 2)  # (B, 64, 3*3*D)
        
        # project to llm space
        img_emb = self.dense(vit_features)

        # --- Prepare for Generation ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb

        placeholder_mask = (tokens == self.img_token_id)
        img_weight_mask = placeholder_mask.unsqueeze(-1).float()

        num_placeholders = placeholder_mask.sum(dim=1)
        if not torch.all(num_placeholders == self.img_seq_len):
            raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                             f"does not match the expected number of image embeddings ({self.img_seq_len}).")

        final_embeddings[placeholder_mask] = img_emb.reshape(-1, self.d_model)

        # --- Autoregressive Generation Loop ---
        generated_outputs = final_embeddings
        newly_generated_ids_list = []

        for i in range(max_new_tokens):
            # Forward pass
            prompt_output = self.decoder(generated_outputs, generated_outputs, attention_mask=None, img_weight_mask=img_weight_mask)
            last_output = prompt_output[:, -1, :] / temp  

            # --- Strict greedy: take argmax instead of sampling ---
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)

            # Append to sequence
            newly_generated_ids_list.append(next_token)
            next_emb = self.decoder.token_embedding(next_token)
            generated_outputs = torch.cat((generated_outputs, next_emb), dim=1)

            zero_mask = torch.zeros(
                img_weight_mask.size(0), 1, img_weight_mask.size(-1),
                device=img_weight_mask.device,
                dtype=img_weight_mask.dtype
            )
            img_weight_mask = torch.cat((img_weight_mask, zero_mask), dim=1)

            # Check for EOS
            if next_token.item() == 2:  # EOS token ID
                break

        return newly_generated_ids_list

    
# Build and load VLM
vlm = VLM(cfg, activate_lora=False)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/multimodal5_dino_pixelshuffle/vlm_trained_weights_stage1.pt',
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

vlm.load_state_dict(new_state_dict, strict=True)
vlm.to(device)
vlm.eval()

def encode(text: str) -> List[int]:
    # Add image tokens
    image_tokens = img_token_id * IMG_SEQ_LEN

    # Encode the first user input9
    user_input = tokenizer.encode(text).ids

    full_ids = system_prompt1 + image_tokens + system_prompt2 + user_input + assistant_prompt
    return full_ids

def generate_answer(model, image_path, question, save_dir, max_tokens=70, temp=1.0):

    mean=(0.485,0.456,0.406)
    std =(0.229,0.224,0.225)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # ---- Load image ----
    imgbgr = cv2.imread(image_path)
    if imgbgr is None:
        print("Image not found:", image_path)
        return

    # ---- Preprocess image ----
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(imgrgb, (384, 384))
    img_tensor = transform(resized).unsqueeze(0).to(device)

    # ---- Encode text ----
    prompt_ids = encode(question)
    text_tensor = torch.tensor([prompt_ids], dtype=torch.int32).to(device)

    # ---- Generate answer ----
    output = model.generate((img_tensor, text_tensor),
                            max_new_tokens=max_tokens, temp=temp)

    decoded = tokenizer.decode(output).strip()
    print("Raw output:", decoded)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(image_path).split('.')[0]

    filename = f"{base}_{decoded[:10]}.png"
    filepath = os.path.join(save_dir, filename)

    plt.imsave(filepath, imgrgb)
    print("Saved:", filepath)

def draw_and_save_bbox(image_rgb, decoded, save_path):
    """
    decoded: model output text containing something like:
             "(0.2, 0.2, 0.6, 0.6)"
    """

    # ---- Find numbers in the output ----
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", decoded)
    if len(nums) < 4:
        print("Could not find bbox in output:", decoded)
        return False

    x1, y1, x2, y2 = map(float, nums[:4])

    H, W, _ = image_rgb.shape

    # ---- Convert normalized to absolute ----
    X1 = int(x1 * W)
    Y1 = int(y1 * H)
    X2 = int(x2 * W)
    Y2 = int(y2 * H)

    # ---- Draw box ----
    img_draw = image_rgb.copy()
    cv2.rectangle(img_draw, (X1, Y1), (X2, Y2), (255, 0, 0), 3)

    # ---- Save ----
    plt.imsave(save_path, img_draw)
    print("BBox saved:", save_path)
    return True

def generate_answer_bbox(model, image_path, question, save_dir, max_tokens=70, temp=0.5):

    mean=(0.485,0.456,0.406)
    std =(0.229,0.224,0.225)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # ---- Load image ----
    imgbgr = cv2.imread(image_path)
    if imgbgr is None:
        print("Image not found:", image_path)
        return

    # ---- Preprocess image ----
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(imgrgb, (384, 384))
    img_tensor = transform(resized).unsqueeze(0).to(device)

    # ---- Encode text ----
    prompt_ids = encode(question)
    text_tensor = torch.tensor([prompt_ids], dtype=torch.int32).to(device)

    # ---- Generate answer ----
    output = model.generate((img_tensor, text_tensor),
                            max_new_tokens=max_tokens, temp=temp)

    decoded = tokenizer.decode(output).strip()
    print("Raw output:", decoded)

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(image_path).split('.')[0]

    # ---------------- Draw bbox and Save -----------------
    filename = f"{base}_bbox.png"
    filepath = os.path.join(save_dir, filename)

    ok = draw_and_save_bbox(imgrgb, decoded, filepath)
    if not ok:
        print("Failed to draw bbox, saving original image instead.")
        plt.imsave(filepath, imgrgb)

    print("Saved:", filepath)


save_dir = "model_inference1_"

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020247.jpg'
# ques = "Find bounding box coordinate of the bear."
# print(f"\n Predicting question: {ques}")
# generate_answer_bbox(vlm, img, ques, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000021903.jpg'
# ques = "Find bounding box coordinate of the person."
# print(f"\n Predicting question: {ques}")
# generate_answer_bbox(vlm, img, ques, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000019432.jpg'
# ques = "Find bounding box coordinate of the person."
# print(f"\n Predicting question: {ques}")
# generate_answer_bbox(vlm, img, ques, save_dir, temp=1.0)


# q = ["Where is the person standing? What is he doing?",
#      'Do you see any vehicle? What is the colour of the vehicle?',
#      "What is the colour of the clothes the person is wearing ?",
#      "Which side of the court is person standing?",
#      "What is the person doing?",
#      'Which ethnicity does the person belong to?',
#      "How many people do you see?",
#      "Is there any animal? If so which animal is it?",
#      'What is going on in the image?',
#      "What is the main focus of the image?",
#      "What might be the benefits of engaging in this outdoor activity?",
#      "What is the key skill required for the game the young man is playing?"]

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000019432.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, temp=1.0)

# q = ['Do you see any vehicle? What is the colour of the vehicle?',
#      "What is the colour of the clothes the person is wearing ?",
#      "What is the person doing?",
#      'Which ethnicity does the person belong to?',
#      "How many people do you see?",
#      "Is there any animal? If so which animal is it?",
#      "What is the main focus of the image?"]

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000021879.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000021903.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, temp=1.0)

# q = ['Do you see any vehicle? What is the colour of the vehicle?',
#      "How many people do you see?",
#      "Describe the image in detail."
#      "Is there any animal? If so which animal is it?",
#      "Is it day or night time?"]

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020247.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, temp=1.0)


# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020553.jpg'
# q = 'Describe the location and surrounding in detail.'
# print(f"\n Predicting question: {q}")
# generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

# q = ['Do you see any vehicle? What is the colour of the vehicle?',
#      "What is the colour of the vehicle?",
#      "How many people do you see?",
#      "Describe the image in detail.",
#      "Is there any animal? If so which animal is it?",
#      "Is it day or night?",
#      "Where is the image taken? Describe the location.",
#      "Describe the background.",
#      "What is the person doing?",
#      "What is the purpose of the man in the image?"]

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000018737.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, temp=1.0)

# q = ['Do you see any vehicle? What is the colour of the vehicle?',
#      "How many people do you see?",
#      "Describe the image in detail.",
#      "Is there any animal? If so which animal is it?",
#      "Is it day or night?",
#      "Where is the image taken? Describe the location.",
#      "Describe the background.",
#      "Describe all the animals present in the image."]

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000578236.jpg'
# for i in q:
#     print(f"\n Predicting question: {i}")
#     generate_answer(vlm, img, i, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000018837.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020247.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020553.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000021903.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000021879.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000019432.jpg'
q = 'Describe the image in detail.'
print(f"\n Predicting question: {q}")
generate_answer(vlm, img, q, save_dir, max_tokens=120, temp=1.0)
