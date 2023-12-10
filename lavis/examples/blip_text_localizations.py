import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from matplotlib import pyplot as plt
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import numpy as np


def evaluate_saliency_map(self, raw_image, model_path, caption, block_num=7, dst_w=720):
    # Setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and preprocessors
    model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", model_path, device=device, is_eval=True)

    raw_image = Image.fromarray(np.uint8(raw_image)).convert('RGB')
    # Plot utilities for GradCam
    w, h = raw_image.size
    scaling_factor = dst_w / w
    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255

    # Preprocess image and text inputs
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)

    # Compute GradCam
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=block_num)

    # Average GradCam for the full image
    avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])

    # GradCam for each token
    num_image = len(txt_tokens.input_ids[0]) - 2
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)

    plt.show()
    return gradcam_image


def main(image_path, model_path, caption, block_num=7, dst_w=720):
    # Load an example image and text
    raw_image = Image.open(image_path).convert("RGB")

    # Setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and preprocessors
    model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", model_path, device=device, is_eval=True)

    # Plot utilities for GradCam
    w, h = raw_image.size
    scaling_factor = dst_w / w
    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255

    # Preprocess image and text inputs
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)

    # Compute GradCam
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=block_num)

    # Average GradCam for the full image
    avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(avg_gradcam)
    ax.set_yticks([])
    ax.set_xticks([])

    # GradCam for each token
    num_image = len(txt_tokens.input_ids[0]) - 2
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)

    plt.show()


if __name__ == "__main__":

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and preprocessors
    model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
