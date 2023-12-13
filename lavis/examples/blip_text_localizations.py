from matplotlib import pyplot as plt
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import numpy as np
from pysaliency.external_datasets.sjtuvis import TextDescriptor
import pysaliency

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from PIL import Image

import sys
import os
# sys.path.append("../..")
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

data_location = "datasets/test"
text_descriptor = TextDescriptor('datasets/test/original_sjtuvis_dataset/text.xlsx')
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_sjtu_vis("datasets/test/original_sjtuvis_dataset", location=data_location, text_descriptor=text_descriptor)

idx = 0
stimulus = Image.fromarray(mit_stimuli[idx].stimulus_data)
filename = os.path.basename(mit_stimuli[idx].filename)
text_description = text_descriptor.get_description(filename)
print(text_description, filename)

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption = "Merlion near marina bay."
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

dst_w = 720
raw_image = stimulus
w, h = raw_image.size
scaling_factor = dst_w / w

resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
norm_img = np.float32(resized_img) / 255

img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)
txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)
# fig, ax = plt.subplots(num_image, 1, figsize=(15,5*num_image))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(avg_gradcam)

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

    # Maximum number of rows and columns per row
max_rows = 10
max_cols_per_row = 5  # Adjust based on desired layout

# Determine the number of tokens to be displayed in each row
num_tokens = len(txt_tokens.input_ids[0]) - 2
tokens_per_row = min(max_cols_per_row - 2, num_tokens)  # Subtract 2 for original and avg_gradcam

gradcam_iter = iter(gradcam[0][2:-1])
token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

for row in range(max_rows):
    # Create a subplot for this row
    fig, axes = plt.subplots(1, tokens_per_row + 2, figsize=(15, 5))  # Adjust figsize as needed

    # Display the original image and avg_gradcam
    axes[0].imshow(norm_img)
    axes[0].set_title("Original Image")
    axes[1].imshow(avg_gradcam)
    axes[1].set_title("Avg GradCAM")

    for col in range(2, tokens_per_row + 2):
        try:
            gradcam, token_id = next(zip(gradcam_iter, token_id_iter))
        except StopIteration:
            break  # No more tokens to display

        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        axes[col].imshow(gradcam_image)
        axes[col].set_title(word)

    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks([])

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove gaps between subplots
    plt.show()

    if row * tokens_per_row >= num_tokens:
        break  # Stop if we have displayed all tokens
