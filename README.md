<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/logo1.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" />
    <img alt="powered by Pytorch" src="https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white">
    <img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" />
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLearner209%2FAugmentIQ&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</p>

# BLIP BeaconSaliency: Harnessing BERT and ViT Synergy for Text-Guided Visual Focus in Deep Learning Models

The official code repository for the paper: **BLIP BeaconSaliency: Harnessing BERT and ViT Synergy for Text-Guided Visual Focus in Deep Learning Models.**

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/teaser.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

## Introduction:

**BLIP BeaconSaliency**, is a text-guided saliency prediction model base on BLIP and GradCam. leverages the synergy between visual-text matching-—a forefront in multi-modal learning—and text-guided saliency models, marks a significant stride in this domain. This repo implements our innovative approach, which showcases exceptional performance across diverse datasets. Our methodology, encapsulates a unique blend of technical prowess and creative ingenuity, setting a new benchmark in the field. The introduction of our work seeks to not only articulate the technical merits of our model but also to engage and inspire the reader, bridging the gap between complex computational concepts and their practical implications in understanding visual attention. Our code and finetuned models can be obtained at.

In essence, **BLIP BeaconSaliency** is more than an addition to the compendium of visual saliency methodologies; it is an approach that aligns with the complexities of visual saliency prediction and intricate image processing technologies.

🤗 Please cite [BLIP BeaconSaliency](https://github.com/Learner209/text-guided-saliency) in your publications if it helps with your work. Please star🌟 this repo to help others notice AugmentIQ if you think it is useful. It really means a lot to our open-source research. Thank you! BTW, you may also like [`BLIP`](https://github.com/salesforce/BLIP), [`pysaliency`](https://github.com/matthias-k/pysaliency), the two great open-source repositories upon which we built our architecture.

> 📣 Attention please: <br> > **BLIP BeaconSaliency** is developed under the framework of [pysaliency](https://github.com/matthias-k/pysaliency), a Python toolbox for predicting visual saliency maps and is pre-configured with several datasets and matlab-implemented models. An example of using pysaliency for predicting SJTU-VIS dataset is shown below. With [pysaliency](https://github.com/matthias-k/pysaliency), easy peasy! 😉

<details open>
  <summary><b>👉 Click here to see the example 👀</b></summary>

```bash
cd pysaliency
pip install -e .
```

```python
# Data preprocessing. Tedious, but pysaliency can help. 🤓
data_location = "../../datasets"
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=data_location) # For datasets in pysaliency database, pysaliency will automatically download, extract it, and store it in hfd5 format.
index = 10
plt.imshow(mit_stimuli.stimuli[index])
f = mit_fixations[mit_fixations.n == index]
plt.scatter(f.x, f.y, color='r')
_ = plt.axis('off')
cutoff = 20

aim: pysaliency.SaliencyMapModel = pysaliency.AIM(location='../../models', cache_location=os.path.join('model_caches', 'AIM'))

from pysaliency.external_datasets.sjtuvis import TextDescriptor
text_descriptor = TextDescriptor('../../datasets/test/original_sjtuvis_dataset/text.xlsx')
data_location = "../../datasets/test"
original_dataset_path = "../../datasets/test/original_sjtuvis_dataset"
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_sjtu_vis(original_dataset_path=original_dataset_path, location=data_location, text_descriptor = text_descriptor)
short_stimuli = pysaliency.FileStimuli(filenames=mit_stimuli.filenames[:cutoff]) # hold first 10 visual stimuli as pilot test set.
short_fixations = mit_fixations[mit_fixations.n < cutoff]

# Model inference. This is pysaliency showtime. 💪
smap = model.saliency_map(mit_stimuli[10])
plt.imshow(smap)
plt.show()


# Several evaluation metrics calculation.
auc_uniform = model.AUC(short_stimuli, short_fixations, nonfixations='uniform', verbose=True)  # Measures the accuracy of predicting fixations by evaluating the model's ability to distinguish between fixated and non-fixated regions.
auc_shuffled = model.AUC(short_stimuli, short_fixations, nonfixations='shuffled', verbose=True)
auc_identical_nonfixations = model.AUC(short_stimuli, short_fixations, nonfixations=short_fixations, verbose=True)
kl_uniform = model.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations='uniform') #  Quantifies the dissimilarity between fixation and nonfixation distributions, providing insights into the model's ability to capture saliency patterns.
kl_shuffled = model.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations='shuffled')
kl_identical = model.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations=short_fixations)
nss = model.NSS(short_stimuli, short_fixations) # Measures the alignment between the model's saliency values and human eye fixations, indicating consistency with human gaze behavior.
gold_standard = pysaliency.FixationMap(short_stimuli, short_fixations, kernel_size=30)
image_based_kl = model.image_based_kl_divergence(short_stimuli, gold_standard) # Calculates the similarity between the model's saliency predictions and the ground truth fixation map using KL divergence.
cc = model.CC(short_stimuli, gold_standard) # Measures the linear relationship between the model's saliency predictions and the ground truth fixation map.
ssim = model.SIM(short_stimuli, gold_standard) # Measures the structural similarity between the model's saliency predictions and the ground truth fixation map.


```

## ❖ Contributions and Performance

⦿ **`Contributions`**:

1.  Our model innovatively explores text-guided saliency, an area previously under-researched. It focuses on how textual prompts affect visual saliency map prediction, advancing the field’s boundaries.
2.  The architecture integrates BERT and GradCam. BERT excels in aligning text and image features, while GradCam effectively draws saliency maps using esti- mated gradient flows. Their combination enhances model performance on various metrics.
3.  Our model achieves better and consistent results, comparable to state-of-the-art models across several metrics like sAUC, AUC, SSIM, CC, confirming its efficacy.

⦿ **`Performance`**: **BLIP BeaconSaliency** outperforms various models on several saliency benchmarks, especially in datset SJTU-VIS.

## ❖ Brief Graphical Illustration of Our Methodology

Here we only show the main component of our method: the joint-optimization training approach combining three encoders while frozening their own weights.
For the detailed description and explanation, please read our full paper if you are interested.

<b>Fig. 1: Training approach</b>

## ❖ Repository Structure

The implementation of **BLIP BeaconSaliency** is in dir [`blipsaliency`](https://github.com/Learner209/text-guided-saliency).Please install it via `pip install -e .` or `python setup.py install`. Due to the time and resource limit, we haven't performed extensive enough parameter finetuning experiments, if you like this repo, plek and PR to help us improve it ! 💚 💛 🤎.

## ❖ Development Environment

We run on `Ubuntu 22.04 LTS` with a system configured with a NVIDIA RTX A40 GPU.

-   Use conda to create a env for **BLIP BeaconSaliency** and activate it.

```bash
conda env create --file=enviornment.yaml
conda activate blipsaliency
```

-   Then install **blipsaliency** as a package

```
cd blipsaliency
pip install -e .
```

Additionally, if you want to reproduce the results about the gradient-based models in the paper(extensive experiments on Vanilla Gradient, SmoothGrad, Integrated gradients, GradCam... based on InceptionV3, CNN or MLP architectures), please refer to the sanity-check directory !🤗

To follow the original implementation of these gradient-based models, we have to create a new conda environment to test it!

```bash
conda env create --file=tf_env.yaml
conda activate sanity
```

Also, if you want to see how I process the SJTU-VIS dataset inot to hdf5 format and integrate them into the pysaliency framework, please refer to the pysaliency directory and more specifically, the file, (and we are planning to give a PR to the original pysaliency repo since SJTU-VIS is an awesome dataset integrated with the power of textual prompts) !😎

## ❖ Datasets

We run on nine datasets, more specifically, MIT1003(MIT300), SJTU-VIS, CAT2000, FIGRIM, SALICON, Toronto, DUT-OMRON, OSIE, PASCAL-S, NUSEF-Public.

Here are some samples taken randomly from the dataset:

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/cat2000.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/FIGRIM.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

Some samples taken from the SJTU-VIS dataset, coupled with multi-scale textual prompts(including non-salient objects, salient objects, personal computersglobal and local context).

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/sjtuvis.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

Now the directory tree should be the following:

```
- datasets
- deepgaze
    - deepgaze_pytorch
- docs
- blipsaliency
    - examples
    - blipsaliency
    - projects
- pysaliency
    - notebooks
    - optpy
    - pysaliency
- models
- pretrained_weights
    - deepgaze
- saliency_toolbox
- sanity_check
    - notebooks
    - src
```

## ❖ Usage

We use the BLIP pretrained model as our back-bone, and finetune it on the SJTU-VIS dataset. Also please take a tour to the [`pysaliency`](https://github.com/matthias-k/pysaliency) repo for further details.

## ❖ Quick Run

<details open>
  <summary><b>👉 Click here to see the example 👀</b></summary>

Please see the `.ipynb` notebooks under the `pysaliency/notebooks/` directory for reference about the training procedure and inference pass of our best model.

Please see the `.ipynb` notebooks under the `sanity_check/notebooks/` directory for experiments about various gradient-based models with their backbone choosing from InceptionV3, CNN, MLP architecture.

</details>

❗️Note that paths of datasets and saving dirs may be different on your PC, please check them in the configuration files.

## ❖ Experimental Results

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/heatmap2.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/super_banner.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

<p align="center">
    <a id="SAITS" href="#SAITS">
        <img src="./docs/_static/super_banner2.png" alt="SAITS Title" title="SAITS Title" width="600"/>
    </a>
</p>

## ❖ Acknowledgments

I extend my heartfelt gratitude to the esteemed faculty and dedicated teaching assistants of CS3324 for their invaluable guidance and support throughout my journey in image processing. Their profound knowledge, coupled with an unwavering commitment to nurturing curiosity and innovation, has been instrumental in my academic and personal growth. I am deeply appreciative of their efforts in creating a stimulating and enriching learning environment, which has significantly contributed to the development of this paper and my understanding of the field. My sincere thanks to each one of them for inspiring and challenging me to reach new heights in my studies.

### ✨Stars/forks/issues/PRs are all welcome!

<details open>
<summary><b><i>👏 Click to View Contributors: </i></b></summary>

![Stargazers repo roster for @Learner209/text-guided-saliency](http://reporoster.com/stars/dark/Learner209/text-guided-saliency)

</details>

## ❖ Last but Not Least

If you have any additional questions or have interests in collaboration,please take a look at [my GitHub profile](https://github.com/Learner209) and feel free to contact me 😃.
