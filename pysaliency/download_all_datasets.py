import pysaliency

dataset_location = '../../datasets'

model_location = '../../models'
try:
    aim = pysaliency.AIM(location=model_location)
except Exception as e:
    print(f"Loading model exception : {e}")
    print("Failed to load the model: AIM !")

# Download the CAT2000 Train dataset
try:
    cat_train_stimuli, cat_train_fixations = pysaliency.external_datasets.get_cat2000_train(location=dataset_location, version='1.1')
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_cat2000_train !")

# Download the iSUN Training dataset
try:
    isun_training_stimuli, isun_training_fixations = pysaliency.external_datasets.get_iSUN_training(location=dataset_location)
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_iSUN_training !")

# Download the iSUN Validation dataset
try:
    isun_validation_stimuli, isun_validation_fixations = pysaliency.external_datasets.get_iSUN_validation(location=dataset_location)
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_iSUN_validation !")

# Download the iSUN Testing dataset
try:
    isun_testing_stimuli, isun_testing_fixations = pysaliency.external_datasets.get_iSUN_testing(location=dataset_location)
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_iSUN_testing !")

# Download the OSIE dataset
try:
    osie_stimuli, osie_fixations = pysaliency.external_datasets.get_OSIE(location=dataset_location)
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_OSIE !")

# Download the NUSEF Public dataset
try:
    nusef_public_stimuli, nusef_public_fixations = pysaliency.external_datasets.get_NUSEF_public(location=dataset_location)
except Exception as e:
    print(f"The exception is : {e}")
    print("Failed to download the dataset: get_NUSEF_public !")
