from llm_lora import *
import timm
import torch
from tokenizers import Tokenizer
from torchvision.transforms import v2 as T
import torchvision.io as io
import json
import pandas as pd
import numpy as np
import os
import ast
from PIL import Image
import re
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import List
from torch.optim.lr_scheduler import _LRScheduler
import math
from tqdm import tqdm
import cv2
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()

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

IMG_SEQ_LEN = 196

class VLM(torch.nn.Module):
    def __init__(self, cfg, activate_lora=False):
        super().__init__()
        # --- Language Model ---
        self.decoder = LanguageModel.from_pretrained(cfg, activate_lora=activate_lora)
        self.d_model = cfg.lm_hidden_dim
        self.img_token_id = img_token_id[0]

        # --- Image Embedding Model (ViT) ---
        # Using a small ViT model as an example
        self.img_emb_model = timm.create_model('deit3_base_patch16_224_in21ft1k',pretrained=False)

        ckpt = "/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3/model_weights/pytorch_model.bin"
        checkpoint = torch.load(ckpt, map_location='cpu')

        # --- Handle various checkpoint structures ---
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # assume checkpoint itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # --- Load state dict ---
        missing, unexpected = self.img_emb_model.load_state_dict(state_dict, strict=False)
        print("✅ Missing keys:", missing)
        print("⚠️ Unexpected keys:", unexpected)

        self.img_feature_dim = 768

        # --- Learnable Query Tokens and Cross-Attention ---
        self.img_seq_len = IMG_SEQ_LEN

        # --- Dense Projector ---
        # This now projects the ViT patch embeddings to the LLM's hidden dimension
        self.dense = torch.nn.Sequential(
                    torch.nn.Linear(self.img_feature_dim, 2*self.d_model),  # expand
                    torch.nn.GELU(),                                          # non-linear activation
                    torch.nn.Linear(2*self.d_model, self.d_model)           # project down to LM dim
                    )


        # Freeze pretrained components
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.img_emb_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        img_tensors, tokens = inputs
        img_tensors = img_tensors.to(tokens.device)

        pad_mask = tokens != pad_id[0]

        # --- Get ViT Features ---
        vit_features = self.img_emb_model.forward_features(img_tensors)[:,1:,:]

        # --- Project ViT features to LLM's dimension ---
        img_emb = self.dense(vit_features)

        # --- Project and Combine Embeddings ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb
    
        placeholder_mask = (tokens == self.img_token_id)

        # Sanity check: Ensure the number of placeholders matches the number of image embeddings
        num_placeholders = placeholder_mask.sum(dim=1)
        if not torch.all(num_placeholders == self.img_seq_len):
            raise ValueError(f"The number of image placeholder tokens in the input ({num_placeholders.tolist()}) "
                             f"does not match the expected number of image embeddings ({self.img_seq_len}).")

        # We replace the embeddings at the masked locations.
        # The shape of `final_embeddings[placeholder_mask]` will be (B * img_seq_len, d_model), 
        # The shape of `img_emb` is (B, img_seq_len, d_model), so we reshape it to match.
        final_embeddings[placeholder_mask] = img_emb.reshape(-1, self.d_model)

        logits = self.decoder(final_embeddings, attention_mask=pad_mask)
        return logits
    
    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=30, temp=1.0):
        img_tensors, tokens = inputs
        img_tensors = img_tensors.to(tokens.device)

        # 1. Get ViT patch embeddings
        vit_features = self.img_emb_model.forward_features(img_tensors)[:,1:,:]
        
        # project to llm space
        img_emb = self.dense(vit_features)

        # --- Prepare for Generation ---
        text_emb = self.decoder.token_embedding(tokens)
        final_embeddings = text_emb

        placeholder_mask = (tokens == self.img_token_id)

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
            prompt_output = self.decoder(generated_outputs, attention_mask=None)
            last_output = prompt_output[:, -1, :] / temp  

            # --- Strict greedy: take argmax instead of sampling ---
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)

            # Append to sequence
            newly_generated_ids_list.append(next_token)
            next_emb = self.decoder.token_embedding(next_token)
            generated_outputs = torch.cat((generated_outputs, next_emb), dim=1)

            # Check for EOS
            if next_token.item() == 2:  # EOS token ID
                break

        return newly_generated_ids_list

prompt = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{IMAGE TOKENS}<|im_end|>\n"
    "<|im_start|>system\n"
    "Analyze the provided information and answer the following question.<|im_end|>\n"
    "<|im_start|>user\n"
    "{TEXT TOKENS}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAX_LEN = IMG_SEQ_LEN + 64 + len(tokenizer.encode(prompt).ids)
MAX_INPUT_LEN = 512

def encode_pair(text_a: str, text_b: str) -> np.ndarray:
    # Add image tokens
    image_tokens = img_token_id * IMG_SEQ_LEN

    # Encode the first user input
    user_input = tokenizer.encode(text_a).ids

    # Encode the second user input
    assistant_response = tokenizer.encode(text_b).ids

    # Combine all parts
    encoded = system_prompt1 + image_tokens + system_prompt2 + user_input + assistant_prompt + sep_id + assistant_response

    # Truncate to maximum length
    return encoded[:MAX_LEN]

def encode_example(text: str, summary: str):
    ids    = encode_pair(text, summary)     
    labels = ids[1:]

    ids = ids + pad_id*(MAX_LEN-len(ids))
    
    labels = labels + eos_id 
    labels = labels + pad_id*(MAX_LEN-len(labels))

    ids    = np.array(ids,dtype=np.int32)
    labels = np.array(labels,dtype=np.int32)

    # find SEP
    SEP_idxs = np.where(labels == sep_id)[0]
    SEP_pos  = int(SEP_idxs[0]) if SEP_idxs.size else len(ids)

    # build base mask: 1 only for positions > sep_pos AND not PAD
    positions = np.arange(len(labels))
    loss_mask = (positions > SEP_pos).astype(np.float32) * (labels != pad_id).astype(np.float32)

    # Remove SEP from ids, labels, and loss_mask
    sep_mask_ids    = (ids != sep_id[0])
    sep_mask_labels = (labels != sep_id[0])
    
    ids       = ids[sep_mask_ids]
    labels    = labels[sep_mask_labels]
    loss_mask = loss_mask[sep_mask_labels]

    return ids, labels.astype(np.int32), loss_mask

KeypointLocationDescription = {
'nose': 'The nose is the central, protruding feature on their face, located just above the upper lip.',
'left eye': 'The left eye is the visual organ on the left side of their face, typically located above the left cheek and beside the nose.',
'right eye': 'The right eye is the visual organ on the right side of their face, typically located above the right cheek and beside the nose.',
'left ear': 'The left ear is the auditory organ on the left side of their head, typically located to the side of the left temple.',
'right ear': 'The right ear is the auditory organ on the right side of their head, typically located to the side of the right temple.',
'left shoulder': 'The left shoulder is the joint connecting the left arm and the torso, typically situated on the upper left side of the chest.',
'right shoulder': 'The right shoulder is the joint connecting the right arm and the torso, typically situated on the upper right side of the chest.',
'left elbow': 'The left elbow is the joint connecting the left upper arm and the left forearm, typically situated in the middle of the left arm, between left shoulder and left wrist.',
'right elbow': 'The right elbow is the joint connecting the right upper arm and the right forearm, typically situated in the middle of the right arm, between right shoulder and right wrist.',
'left wrist': 'The left wrist is the joint connecting the left forearm and the left hand, typically located at the base of the left hand.',
'right wrist': 'The right wrist is the joint connecting the right forearm and the right hand, typically located at the base of the right hand.',
'left hip': 'The left hip is the joint connecting the left thigh to the pelvis, typically located on the left side of the lower torso.',
'right hip': 'The right hip is the joint connecting the right thigh to the pelvis, typically located on the right side of the lower torso.',
'left knee': 'The left knee is the joint connecting the left thigh and the left lower leg, typically situated in the middle of the left leg, it is located between the left hip and left ankle.',
'right knee': 'The right knee is the joint connecting the upper leg and lower leg on the right side, it is located between the right hip and right ankle.',
'left ankle': 'The left ankle is the joint connecting the left lower leg and the left foot, typically located at the base of the left leg.',
'right ankle': 'The right ankle is the joint connecting the right lower leg and the right foot, typically located at the base of the right leg.',
}

# --- COCO 17 keypoints ---
COCO_KEYPOINTS = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle"
]

def square_crop_with_margin(img, x, y, w, h, margin=20):
    H, W = img.shape[:2]

    # Expand bbox
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)

    bw = x1 - x0
    bh = y1 - y0

    # square side limited by image bounds
    side = max(bw, bh)
    side = min(side, min(W, H))

    # center
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    sq_x0 = cx - side // 2
    sq_y0 = cy - side // 2

    # clamp
    sq_x0 = max(0, min(sq_x0, W - side))
    sq_y0 = max(0, min(sq_y0, H - side))

    sq_x1 = sq_x0 + side
    sq_y1 = sq_y0 + side

    crop = img[sq_y0:sq_y1, sq_x0:sq_x1]

    return crop, sq_x0, sq_y0

class COCOKeypointVLMDataset(Dataset):
    def __init__(self, coco_json, img_dir, tokenizer, size=224, max_samples=None):
        with open(coco_json, "r") as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.size = size

        mean = (0.485,0.456,0.406)
        std  = (0.229,0.224,0.225)
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        self.samples = []

        # --- Build all (image, keypoint) pairs with tqdm ---
        print("🔍 Building COCO keypoint dataset...")
        for ann in tqdm(self.data["annotations"], desc="Processing annotations"):
            if ann["category_id"] != 1 or ann["num_keypoints"] == 0:
                continue

            img_info = next((img for img in self.data["images"] if img["id"] == ann["image_id"]), None)
            if img_info is None:
                continue

            img_path = os.path.join(self.img_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue  # Skip missing images

            bbox = ann["bbox"]
            keypoints = ann["keypoints"]

            x, y, w, h = bbox
            for i, name in enumerate(COCO_KEYPOINTS):
                xk, yk, v = keypoints[3 * i : 3 * i + 3]
                if v > 0:
                    self.samples.append({
                        "img_path": img_path,
                        "bbox": bbox,
                        "kp_name": name,
                        "kp_coords": (xk, yk)
                    })
            
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"✅ Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad and square
        x, y, w, h = map(int, s["bbox"])
        crop, sx, sy = square_crop_with_margin(img, x, y, w, h, margin=np.random.randint(0,10))

        # keypoint adjusted inside padded crop
        kx, ky = s["kp_coords"]
        px = kx - sx
        py = ky - sy

        # Resize after rotation
        oh, ow = crop.shape[:2]
        crop = cv2.resize(crop, (self.size, self.size))
        crop = self.transform(crop)

        # Scale keypoint coordinates into [0,100] relative to crop
        scaled_x = int(round((px / ow),2)*100)
        scaled_y = int(round((py / oh),2)*100)

        target = f"{scaled_x:02d},{scaled_y:02d}"

        # Construct text prompt
        text = f"{KeypointLocationDescription[s['kp_name']]} Find the coordinates of {s['kp_name']} of this person from the image."

        # Encode text using your encode_example logic
        ids, labels, loss_mask = encode_example(text, target)

        # Convert to tensors
        ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)

        return {
            "image": crop,
            "input_ids": ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }

# Example 
dataset = COCOKeypointVLMDataset( coco_json="/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_train2017.json",
                                  img_dir="/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_train2017", 
                                  tokenizer=tokenizer)

train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    sampler=train_sampler,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)

del(dataset)

vocab = tokenizer.get_vocab()           
id_to_token = {idx: tok for tok, idx in vocab.items()}

# Build and load VLM
vlm = VLM(cfg, activate_lora=True).to(device)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3_/vlm_trained_weights_stage1.pt',
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

vlm.load_state_dict(new_state_dict, strict=True)

for name, param in vlm.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

vlm = torch.nn.parallel.DistributedDataParallel(
    vlm,
    device_ids=[local_rank],
    output_device=local_rank,
)

print("Trainable Parameters:")
print("-" * 60)
total_params = 0
for name, param in vlm.named_parameters():
    if param.requires_grad:
        param_count = param.numel()
        print(f"Parameter: {name:<50} Shape: {str(param.shape):<20} Parameters: {param_count}")
        total_params += param_count
print("-" * 60)
print(f"Total Trainable Parameters: {total_params:,}")

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps_ratio, min_lr, max_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_steps_ratio)
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (current_step / self.warmup_steps)
        else:
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay_factor = math.exp(-1.0 * progress)
            lr = self.min_lr + (self.max_lr - self.min_lr) * decay_factor
            lr = max(lr, self.min_lr)
        return [lr for _ in self.base_lrs]
    
EPOCHS = 10
LEARNING_RATE_MIN = 1e-6
LEARNING_RATE_MAX = 1e-4
WARMUP_RATIO = 0.1

# --- Optimizer ---
optimizer = torch.optim.Adam(vlm.parameters(), lr=LEARNING_RATE_MAX, betas=(0.9, 0.98), eps=1e-9)
criterion = torch.nn.CrossEntropyLoss(reduction='none')  

# --- Calculate total steps and initialize scheduler ---
total_steps = len(dataloader) * EPOCHS
scheduler = CustomLRScheduler(
    optimizer,
    total_steps=total_steps,
    warmup_steps_ratio=WARMUP_RATIO,
    min_lr=LEARNING_RATE_MIN,
    max_lr=LEARNING_RATE_MAX
)

for epoch in range(1, EPOCHS + 1):
    train_sampler.set_epoch(epoch)
    vlm.train()
    tot_loss = tot_correct = tot_tokens = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True, disable=(local_rank != 0))

    for batch in pbar:
        img = batch['image']
        ids = batch["input_ids"].to(device)          # (B, T)
        tgt = batch["labels"].to(device)             # (B, T)
        mask = batch["loss_mask"].to(device)         # (B, T)

        optimizer.zero_grad()
        logits = vlm((img,ids))                          # (B, T, V)

        # ---- loss ----
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))  # (B*T,)
        masked_loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)  # Apply mask and average
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # ---- metrics (no grad) ----
        with torch.no_grad():
            preds = logits.argmax(-1)                # (B, T)
            valid = (mask == 1.0)                    # bool mask where loss_mask is 1.0
            correct = (preds == tgt) & valid         # Correct predictions where mask is 1.0
            n_tok = valid.sum().item()               # Count of masked tokens

            tot_correct += correct.sum().item()
            tot_tokens += n_tok
            tot_loss += masked_loss.item() * n_tok   # Scale by number of valid tokens

        # Update pbar (Only Rank 0 updates)
        if local_rank == 0:
            pbar.set_postfix(
                loss=f"{tot_loss / tot_tokens:.4f}" if tot_tokens > 0 else "N/A",
                acc=f"{tot_correct / tot_tokens:.4f}" if tot_tokens > 0 else "N/A",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

    # --- SAVING SECTION (CRITICAL FIXES) ---

    # 1. Synchronization: Wait for ALL GPUs to reach this line
    dist.barrier()

    # 2. Rank Check: Only GPU 0 is allowed to write to disk
    if local_rank == 0:
        save_path = "vlm_trained_weights_stage2.pt"
        
        print("Saving model...")
        
        # 3. Unwrap DDP: Access .module to get the original model
        # This ensures keys are 'dense.weight', not 'module.dense.weight'
        state_dict_to_save = vlm.module.state_dict()
        
        # OPTIONAL: Filter to save ONLY trainable weights to save space?
        # If you want to save EVERYTHING, just use state_dict_to_save.
        # If you only want the projector (since LLM/ViT are frozen/standard):
        # state_dict_to_save = {k: v for k, v in vlm.module.state_dict().items() if v.requires_grad}
        
        torch.save(state_dict_to_save, save_path)
        print(f"\n✅ Training completed. Weights saved correctly to: {save_path}")

# 4. Cleanup
dist.destroy_process_group()
