import pysaliency
import matplotlib.pyplot as plt

dataset_location = '../../datasets'
model_location = '../../models'

print("Loading MIT1003 dataset")
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=dataset_location)
print("Loading MIT300 dataset")
aim = pysaliency.AIM(location=model_location)
print("Loading AIM model")
saliency_map = aim.saliency_map(mit_stimuli.stimuli[0])
print("Generating saliency map")
plt.imshow(saliency_map)
plt.show()

print("Generating AUC")
auc = aim.AUC(mit_stimuli, mit_fixations)


# my_model = pysaliency.SaliencyMapModelFromDirectory(mit_stimuli, '/path/to/my/saliency_maps')
# auc = my_model.AUC(mit_stimuli, mit_fixations)
