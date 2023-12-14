import os
import pysaliency
from pysaliency.external_models import AIM, SUN, GBVSIttiKoch, Judd, IttiKoch, CovSal
from pysaliency.external_datasets.sjtuvis import TextDescriptor
import numpy as np
import matplotlib.pyplot as plt
from pysaliency.saliency_map_conversion import optimize_for_information_gain

from pysaliency.external_datasets import get_mit1003, get_sjtu_vis, get_cat2000_train, get_FIGRIM, get_mit300, get_mit1003, get_mit1003_onesize, get_SALICON, get_toronto, get_DUT_OMRON, get_OSIE, get_PASCAL_S, get_NUSEF_public


import sys
import pysaliency
from pysaliency.external_datasets.sjtuvis import TextDescriptor

data_location = "datasets/test"
text_descriptor = TextDescriptor('datasets/test/original_sjtuvis_dataset/text.xlsx')
print(text_descriptor.get_description('000000020777_2.png')) 
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_sjtu_vis("datasets/test/original_sjtuvis_dataset", location=data_location, text_descriptor=text_descriptor)

import numpy as np
from pysaliency.datasets import Stimulus, Fixations,  StimuliStimulus

def handle_stimulus(stimulus):
    """
    Make sure that a stimulus is a `Stimulus`-object
    """
    if not isinstance(stimulus, Stimulus):
        stimulus = Stimulus(stimulus)
    return stimulus

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch

DEVICE = 'cuda'


DATASET_MAPPINGS = {
	"mit1003": get_mit1003,
 	"mit1003_onesize": get_mit1003_onesize,
 	"sjtuvis": get_sjtu_vis,
	# "cat2000_train": get_cat2000_train,
	"figrim": get_FIGRIM,
	"salicon_eval": get_SALICON,
	"toronto": get_toronto,
 	"DUT_OMRON": get_DUT_OMRON,
	"OSIE": get_OSIE,
	"PASCAL_S": get_PASCAL_S,
	"NUSEF_public": get_NUSEF_public
}

import pandas as pd
model_zoos = [ "IttiKoch","Judd", "AIM", "SUN", "CovSal"]
columns = ['Model', 'Dataset', 'AUC_shuffled', 'AUC_uniform', 'KL_uniform', 'KL_shuffled', 'KL_identical_nonfixations', 'Image_based_KL_divergence']
results_df = pd.DataFrame(columns=columns)


for model_name in model_zoos:
	print(f"------------------------    Model: {model_name} -----------------------------------")
	model: pysaliency.SaliencyMapModel
	# model_location = "../scripts/models/"
	if model_name == "GoldStandard":
		raise NotImplementedError
	elif model_name == "IttiKoch" or model_name == "Judd":
		print("Loading the model from the pysaliency package")
		model: pysaliency.SaliencyMapModel = eval(model_name)(location = "models/", saliency_toolbox_archive = "SaliencyToolbox.zip",  cache_location=os.path.join('models/model_caches/', model_name), caching=True)
	else:
		model: pysaliency.SaliencyMapModel = eval(model_name)(location = "models/", cache_location=os.path.join('models/model_caches/', model_name), caching=True)
	
	
	
	for dataset_key, dataset_func in DATASET_MAPPINGS.items():
		dataset_location = "datasets/"
		print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{dataset_key}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		if dataset_key == "sjtuvis":
			text_descriptor = TextDescriptor('datasets/test/original_sjtuvis_dataset/text.xlsx')
			mit_stimuli, mit_fixations = dataset_func(original_dataset_path="datasets/test/original_sjtuvis_dataset", location=dataset_location, text_descriptor=text_descriptor)
		elif dataset_key == "cat2000_train":
			mit_stimuli, mit_fixations = dataset_func(location=dataset_location, version = "1.1")
		elif dataset_key == "salicon_train":
			stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val = dataset_func(location=dataset_location)
			mit_stimuli, mit_fixations = stimuli_train, fixations_train
		elif dataset_key == "salicon_eval":
			stimuli_train, stimuli_val, stimuli_test, fixations_train, fixations_val = dataset_func(location=dataset_location)
			mit_stimuli, mit_fixations = stimuli_val, fixations_val
		else:
			mit_stimuli, mit_fixations = dataset_func(location=dataset_location)
			
		cutoff = 10
		short_stimuli = pysaliency.FileStimuli(filenames=mit_stimuli.filenames[:cutoff])
		short_fixations = mit_fixations[mit_fixations.n < cutoff]
		
		try:
			auc_uniform = model.AUC(short_stimuli, short_fixations, nonfixations='uniform', verbose=True)
		except Exception as e:
			auc_uniform = np.nan
		try:
			auc_shuffled = model.AUC(short_stimuli, short_fixations, nonfixations='shuffled', verbose=True)
		except Exception as e:
			auc_shuffled = np.nan
		try:
			auc_identical_nonfixations = model.AUC(short_stimuli, short_fixations, nonfixations=short_fixations, verbose=True)
		except Exception as e:
			auc_identical_nonfixations = np.nan
		try:
			kl_uniform = model.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations='uniform')
		except Exception as e:
			kl_uniform = np.nan
			
		try:
			kl_identical = model.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations=short_fixations)
		except Exception as e:
			kl_identical = np.nan
		try:
			nss = model.NSS(short_stimuli, short_fixations)
		except Exception as e:
			nss = np.nan
		try:
			gold_standard = pysaliency.FixationMap(short_stimuli, short_fixations, kernel_size=30)
			image_based_kl = model.image_based_kl_divergence(short_stimuli, gold_standard)
		except Exception as e:
			image_based_kl = np.nan	
		try:
			cc = model.CC(short_stimuli, gold_standard)
		except Exception as e:
			cc = np.nan
		try:
			ssim = model.SIM(short_stimuli, gold_standard)
		except Exception as e:
			ssim = np.nan

		result = {
			'Model': model_name,
			'Dataset': dataset_key,
			'AUC_shuffled': auc_shuffled,
			'AUC_uniform': auc_uniform,
			'NSS': nss,
			'KL_uniform': kl_uniform,	
			# 'KL_shuffled': kl_shuffled,
			'KL_identical_nonfixations': kl_identical,
			'Image_based_KL_divergence': image_based_kl,
			"cc": cc,
   			"ssim": ssim
		}
		results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
		
  