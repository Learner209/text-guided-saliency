from __future__ import absolute_import, print_function, division

import zipfile
import os
import glob

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from pysaliency.datasets import Fixations, read_hdf5, clip_out_of_stimulus_fixations
from pysaliency.utils import (
    atomic_directory_setup)

import random
import zipfile
import os
import glob

import numpy as np
from natsort import natsorted
from PIL import Image

from pysaliency.datasets import FixationTrains
from pysaliency.utils import (
    TemporaryDirectory,
    filter_files,
    run_matlab_cmd,
    download_and_check,
    atomic_directory_setup,
    build_padded_2d_array)
import shutil
from loguru import logger
from pysaliency.external_datasets.utils import create_stimuli, _load
from loguru import logger
import pandas as pd
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional, Any, Callable, Iterable, Sequence

DEBUG_VIS_DATASET_LOADING = False  # Set this to False to disable debugging visualizations


class TextDescriptor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_text_descriptions()

    def load_text_descriptions(self):
        overall_text_df = pd.read_excel(self.file_path, sheet_name="overall")
        partial_text_df = pd.read_excel(self.file_path, sheet_name="partial")

        self.overall_description_dict: Dict(int, str) = dict(zip(overall_text_df['image'], overall_text_df['text']))
        self.partial_description_dict = {}
        for _, row in partial_text_df.iterrows():
            self.partial_description_dict[(row['image'], row['description'])] = row['text']
        # print(self.overall_description_dict)
        # print(self.partial_description_dict)

    def get_description(self, image_path: str) -> str:
        image_path = os.path.splitext(image_path)[0]
        image_parts = image_path.split('_')
        # logger.info(f"image_path:{image_path} | image_parts:{image_parts}")
        image_number = int(image_parts[0])
        suffix = image_parts[-1]
        res = None
        if len(image_parts) == 1 or suffix == '0':
            # logger.info(f"text:{self.overall_description_dict.get(image_number)}")
            suffix = '1'
            res = self.overall_description_dict.get(image_number)
        if res is not None:
            return res

        # logger.info("text:{}".format(self.partial_description_dict.get((image_number, int(suffix)))))
        return self.partial_description_dict.get((image_number, int(suffix)))

    def update_description(self, image_number, description_number, new_text):
        if (image_number, description_number) in self.partial_description_dict:
            self.partial_description_dict[(image_number, description_number)] = new_text
            return True
        return False

    def add_description(self, image_number, description_number, text):
        self.partial_description_dict[(image_number, description_number)] = text

    def remove_description(self, image_number, description_number):
        if (image_number, description_number) in self.partial_description_dict:
            del self.partial_description_dict[(image_number, description_number)]
            return True
        return False

    def get_data_length(self):
        return len(self.partial_description_dict)


def _get_sjtu_vis(original_dataset_path, dataset_name, text_descriptor, location=None, include_initial_fixation=False, only_1024_by_768=False, replace_initial_invalid_fixations=False):
    assert os.path.exists(original_dataset_path)
    if location:
        location = os.path.join(location, dataset_name)
        if os.path.exists(location) and os.path.exists(os.path.join(location, 'stimuli.hdf5')) and os.path.exists(os.path.join(location, 'fixations.hdf5')):
            stimuli = _load(os.path.join(location, 'stimuli.hdf5'))
            fixations = _load(os.path.join(location, 'fixations.hdf5'))
            return stimuli, fixations

        if not os.path.exists(location):
            os.makedirs(location)

        print('Creating sjtu vis dataset stimuli')
        temp_dir = original_dataset_path

        stimuli_src_location = os.path.join(temp_dir, 'image')
        stimuli_target_location = os.path.join(location, 'stimuli') if location else None
        images = glob.glob(os.path.join(stimuli_src_location, '*.png'))
        images = [os.path.split(img)[1] for img in images]
        stimuli_filenames = natsorted(images)

        if os.path.exists(stimuli_target_location):
            shutil.rmtree(stimuli_target_location)
        # print("The base path is {}".format(stimuli_target_location))
        # print(stimuli_filenames[:10])
        stimuli = create_stimuli(stimuli_src_location, stimuli_filenames, stimuli_target_location)

        subjects = [str(i) for i in range(15)]
        out_path = 'extracted'
        if os.path.exists(os.path.join(temp_dir, out_path)):
            shutil.rmtree(os.path.join(temp_dir, out_path))
        os.makedirs(os.path.join(temp_dir, out_path))

        xs = []
        ys = []
        ts = []
        ns = []
        train_subjects = []
        duration_hist = []
        train_durations = []
        for n, stimulus in enumerate(stimuli_filenames):
            stimulus_size = stimuli.sizes[n]
            height, width = stimulus_size
            # print('Processing stimulus {0}/{1}, file:{2}'.format(n, len(stimuli_filenames), stimulus))

            subject_id = random.randint(0, len(subjects) - 1)
            subject_name = subjects[subject_id]

            stimulus_size = stimuli.sizes[n]
            height, width = stimulus_size

            fixation_image_path = os.path.splitext(stimulus)[0]
            fixation_image_path = os.path.join(os.path.join(temp_dir, "fixation"), stimulus)  # replace with actual directory
            fixation_image = Image.open(fixation_image_path).convert('L')  # Convert to grayscale
            original_img = Image.open(os.path.join(stimuli_src_location, stimulus))

            fixation_image_size = fixation_image.size
            original_img_size = original_img.size
            assert fixation_image_size == original_img_size, "The images have different shapes."

            # Get the text
            image_text = text_descriptor.get_description(stimulus)
            fixation_pixels = np.array(fixation_image)

            # Extract coordinates of fixation points (white pixels)
            res = np.where(fixation_pixels == 255)
            _ys, _xs = res[0:]

            # Generating random timestamps and durations
            num_fixations = len(_xs)
            _ts = np.random.rand(num_fixations) * 3  # Random timestamps between 0 and 3 seconds
            _durations = np.random.rand(num_fixations) * 0.5  # Random durations between 0 and 0.5 seconds

            _xs = _xs.astype(np.float64)
            _ys = _ys.astype(np.float64)
            _ts = _ts.astype(np.float64)
            _durations = _durations.astype(np.float64)
            xs.append(_xs)
            ys.append(_ys)
            ts.append(_ts)
            ns.append(n)
            train_subjects.append(subject_id)
            train_durations.append(_durations)

            # Debugging visualization
            if DEBUG_VIS_DATASET_LOADING:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 5))

                # Show original image
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.title('Original Image')
                plt.axis('off')

                # Show fixation points on the original image
                plt.subplot(1, 2, 2)
                plt.imshow(original_img)
                plt.scatter(_xs, _ys, c='red', s=10)  # Red dots for fixation points
                plt.title('Fixation Points')
                plt.figtext(0.5, 0.01, image_text, wrap=True, horizontalalignment='center', fontsize=12)
                plt.axis('off')
                plt.show()

            try:
                pass
                # logger.info("Subject {0}/{1}: {2} fixations".format(subject_id, len(subjects), len(_xs)))
                # print(f"data: {_xs.shape}, {_ys.shape}, {_ts.shape}, {_durations.shape}")
                # print(f"data: {_xs.min()}, {_xs.max()}, {_ys.min()}, {_ys.max()}, {_ts.min()}, {_ts.max()}, {_durations.min()}, {_durations.max()}")
                # print(_xs[:10])
                # print(_ys[:10])
                # print(_ts[:10])
                # print(n)
                # print(subject_id)
                # print(_durations[:10])
            except:
                pass

            for i in range(len(_durations)):
                duration_hist.append(_durations[:i])

        attributes = {
            'duration_hist': build_padded_2d_array(duration_hist),
        }
        scanpath_attributes = {
            'train_durations': build_padded_2d_array(train_durations),
        }
        fixations = FixationTrains.from_fixation_trains(xs, ys, ts, ns, train_subjects, attributes=attributes, scanpath_attributes=scanpath_attributes)

        if location:
            stimuli.to_hdf5(os.path.join(location, 'stimuli.hdf5'))
            fixations.to_hdf5(os.path.join(location, 'fixations.hdf5'))
    return stimuli, fixations


def get_sjtu_vis(original_dataset_path, location, text_descriptor):
    return _get_sjtu_vis(original_dataset_path, "sjtuvis", location=location, text_descriptor=text_descriptor, include_initial_fixation=False, only_1024_by_768=False, replace_initial_invalid_fixations=False)


if __name__ == '__main__':
    import pysaliency
    import matplotlib.pyplot as plt
    import random

    data_location = "datasets/test"
    text_descriptor = TextDescriptor('datasets/test/original_sjtuvis_dataset/text.xlsx')
    mit_stimuli, mit_fixations = pysaliency.external_datasets.get_sjtu_vis("datasets/test/original_sjtuvis_dataset", location=data_location, text_descriptor=text_descriptor)
    num_rows = 4
    num_cols = 5

    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Iterate over the images and fixations
    for i, ax in enumerate(axes.flat):
        # Get the image and fixations for the current index
        index = random.randint(0, len(mit_stimuli.stimuli) - 1)
        image = mit_stimuli.stimuli[index]
        fixations = mit_fixations[mit_fixations.n == index]
        # Plot the image
        ax.imshow(image)
        ax.axis('off')

        # Plot the fixations as red dots
        ax.scatter(fixations.x, fixations.y, color='r')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
