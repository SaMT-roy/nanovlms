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
    
# Build and load VLM
vlm = VLM(cfg, activate_lora=False)
state_dict = torch.load(
    '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm3_/vlm_trained_weights_stage1.pt',
    map_location=device
)

new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

vlm.load_state_dict(new_state_dict, strict=True)
vlm.to(device)

def encode(text: str) -> List[int]:
    # Add image tokens
    image_tokens = img_token_id * IMG_SEQ_LEN

    # Encode the first user input9
    user_input = tokenizer.encode(text).ids

    full_ids = system_prompt1 + image_tokens + system_prompt2 + user_input + assistant_prompt
    return full_ids

def generate_answer(model, image_path, question, kpt_name, save_dir, max_tokens=30, temp=0.5):
    mean=(0.485,0.456,0.406)
    std=(0.229,0.224,0.225)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    imgbgr = cv2.imread(image_path)
    if imgbgr is None:
        print("⚠️ Could not load image:", image_path)
        return
    
    # Keep a copy for plotting
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    H, W = imgrgb.shape[:2]

    # Prepare model input (224x224)
    imgrgb_resized = cv2.resize(imgrgb, (224, 224))
    img_tensor = transform(imgrgb_resized).unsqueeze(0)

    # Encode question → tokens
    prompt_ids = encode(question)
    text_tokens = torch.tensor([prompt_ids], dtype=torch.int32).to(device)

    # Model inference
    output = model.generate((img_tensor, text_tokens), max_new_tokens=max_tokens, temp=temp)
    coords_text = tokenizer.decode(output).strip()
    print("Raw model output:", coords_text)

    # Parse coordinates
    try:
        coords = ast.literal_eval(coords_text)
        x_norm, y_norm = coords
    except Exception as e:
        print("⚠️ Failed to parse coordinates:", coords_text, e)
        return

    # Convert to pixel coordinates (original image scale)
    x = int(x_norm * W)
    y = int(y_norm * H)

    # Draw the keypoint
    im_plot = imgrgb.copy()
    cv2.circle(im_plot, (x, y), 3, (255, 0, 0), -1)  # red dot

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(image_path)       
    name_no_ext = os.path.splitext(base_name)[0]   
    filename = f"{name_no_ext}_{kpt_name}.png"
    filepath = os.path.join(save_dir, filename)

    # Save the plot
    plt.imshow(im_plot)
    plt.title(f"{kpt_name}: ({x}, {y})")
    plt.axis("off")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close()    # prevent memory leak when running many predictions

    print(f"Saved: {filepath}")

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

def get_cropped_bboxes(img_path, annotations_path):
    """
    Reads an image and its COCO annotations, then crops bounding boxes.

    Args:
        img_path (str): Path to the image file.
        annotations_path (str): Path to COCO annotations JSON.
        save (bool): If True, save cropped images.
        save_dir (str): Directory to save crops if save=True.

    Returns:
        list of np.ndarray: List of cropped images as numpy arrays.
    """
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    # Load annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Extract image id
    image_filename = os.path.basename(img_path)
    image_info = next((img for img in coco_data["images"] if img["file_name"] == image_filename), None)
    if image_info is None:
        raise ValueError(f"Image {image_filename} not found in annotation file.")

    image_id = image_info["id"]

    # Filter annotations for this image
    image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

    cropped_images = []
    for i, ann in enumerate(image_annotations):
        x, y, w, h = map(int, ann["bbox"])
        crop, sx, sy = square_crop_with_margin(img, x, y, w, h, margin=1)
        cropped_images.append(crop)

    return cropped_images

def generate_answer_cocoval(model, image_path, annotations_path, question, kpt_name, save_dir, max_tokens=30, temp=0.5):
    mean=(0.485,0.456,0.406)
    std=(0.229,0.224,0.225)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    images = get_cropped_bboxes(image_path, annotations_path)

    for i in range(len(images)):

        imgbgr = images[i]

        if imgbgr is None:
            print("⚠️ Could not load image:", image_path)
            return
        
        # Keep a copy for plotting
        imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
        H, W = imgrgb.shape[:2]

        # Prepare model input (224x224)
        imgrgb_resized = cv2.resize(imgrgb, (224, 224))
        img_tensor = transform(imgrgb_resized).unsqueeze(0)

        # Encode question → tokens
        prompt_ids = encode(question)
        text_tokens = torch.tensor([prompt_ids], dtype=torch.int32).to(device)

        # Model inference
        output = model.generate((img_tensor, text_tokens), max_new_tokens=max_tokens, temp=temp)
        coords_text = tokenizer.decode(output).strip()
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", coords_text)
        print("Raw model output:", coords_text, matches)

        if len(matches) >= 2:
            x_norm, y_norm = map(float, matches[:2])
        else:
            print("⚠️ Could not extract two coordinates:", coords_text)
            return

        # Convert to pixel coordinates (original image scale)
        x = int(x_norm * W / 100)
        y = int(y_norm * H / 100)

        print(x,y,W,H)

        # Draw the keypoint
        im_plot = imgrgb.copy()
        cv2.circle(im_plot, (x, y), 3, (255, 0, 0), -1)  # red dot

        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_path)       
        name_no_ext = os.path.splitext(base_name)[0]   
        filename = f"{name_no_ext}_{kpt_name}.png"
        filepath = os.path.join(save_dir, filename)

        plt.imsave(filepath, im_plot)
        print(f"Saved: {filepath}")

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

save_dir = "model_inference_stage1"

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/person2.png'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     generate_answer(vlm, img, ques, kpt, save_dir, temp=1.0)

img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000019432.jpg'
ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_val2017.json'
for kpt, desc in KeypointLocationDescription.items():
    ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
    print(f"\n Predicting: {kpt}")
    generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000020333.jpg'
# ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_val2017.json'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000019924.jpg'
# ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_val2017.json'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_train2017/000000000294.jpg'
# ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_train2017.json'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000018519.jpg'
# ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_val2017.json'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000022371.jpg'
# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_nanovlm/nanovlm/flip_000000022371.jpg'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer(vlm, img, ques, kpt, save_dir, temp=1.0)

# img = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_val2017/000000023126.jpg'
# ann = '/pfs01/performance-tier/rd_ici/algo_train/saptamtdir/saptamt/locllm_author/coco_keypoints_val2017.json'
# for kpt, desc in KeypointLocationDescription.items():
#     ques = f"{desc} Find the coordinates of {kpt} of this person from the image."
#     print(f"\n Predicting: {kpt}")
#     generate_answer_cocoval(vlm, img, ann, ques, kpt, save_dir, temp=1.0)
