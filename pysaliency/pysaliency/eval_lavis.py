from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from matplotlib import pyplot as plt
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import numpy as np
from pysaliency.datasets import Stimulus, Fixations, FileStimuli, create_subset, FixationTrains, StimuliStimulus
from torch import optim
from tqdm import tqdm
import sys
import pysaliency
from pysaliency.external_datasets.sjtuvis import TextDescriptor
from loguru import logger
import time
import torch.nn as nn
import argparse
from argparse import Namespace
import random
from typing import Optional, Union, List, Tuple, Dict, Any
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_image_text_matching import BlipITM


def handle_stimulus(stimulus: Stimulus):
    """
    Make sure that a stimulus is a `Stimulus`-object
    """
    if not isinstance(stimulus, Stimulus):
        stimulus = Stimulus(stimulus)
    return stimulus


class SaliencyDatasetLoader:
    def __init__(self, stimuli: FileStimuli, fixations: FixationTrains):
        self.stimuli = stimuli
        self.fixations = fixations
        self.split_dataset()

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        # Assuming that stimuli and fixations are aligned and have the same length
        total_stimuli = len(self.stimuli)
        indices = np.arange(total_stimuli)
        np.random.shuffle(indices)

        train_end = int(total_stimuli * train_ratio)
        val_end = train_end + int(total_stimuli * val_ratio)

        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]

    def get_train_data(self):
        return create_subset(self.stimuli, self.fixations, self.train_indices.tolist())

    def get_val_data(self):
        return create_subset(self.stimuli, self.fixations, self.val_indices.tolist())

    def get_test_data(self):
        return create_subset(self.stimuli, self.fixations, self.test_indices.tolist())


class MySaliencyMapModel(pysaliency.SaliencyMapModel):

    def __init__(self, text_descriptor: TextDescriptor, args: Namespace = None, caching: bool = True):
        super(MySaliencyMapModel, self).__init__()
        self.text_descriptor = text_descriptor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "large"
        self.args = args
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip_image_text_matching", model_path, device=self.device, is_eval=True)
        # for parameter in self.model.text_encoder.parameters():
        #     parameter.requires_grad = False
        # for parameter in self.model.visual_encoder.parameters():
        #     parameter.requires_grad = False

    def saliency_map(self, stimulus: Stimulus):
        """
        Get saliency map for given stimulus.

        To overwrite this function, overwrite `_saliency_map` as otherwise
        the caching mechanism is disabled.
        """
        stimulus = handle_stimulus(stimulus)
        stimulus_filename: str = stimulus.filename
        text_description = self.text_descriptor.get_description(os.path.basename(stimulus_filename))

        if not self.caching:
            return self._saliency_map(stimulus.stimulus_data, text_description=text_description)
        stimulus_id: int = stimulus.stimulus_id
        if not stimulus_id in self._cache:
            self._cache[stimulus_id] = self._saliency_map(stimulus.stimulus_data, text_description=text_description)
        return self._cache[stimulus_id]

    def _saliency_map(self, stimulus: np.ndarray, text_description: str):
        return self.evaluate_avg_saliency_map(stimulus[..., :3], text_description)

    def train_mode(self):
        for parameter in self.model.vision_proj.parameters():
            parameter.requires_grad = True
        for parameter in self.model.text_proj.parameters():
            parameter.requires_grad = True
        for parameter in self.model.itm_head.parameters():
            parameter.requires_grad = True

    def eval_mode(self):
        for parameter in self.model.vision_proj.parameters():
            parameter.requires_grad = False
        for parameter in self.model.text_proj.parameters():
            parameter.requires_grad = False
        for parameter in self.model.itm_head.parameters():
            parameter.requires_grad = False

    def evaluate_avg_saliency_map(self, raw_image: np.ndarray, caption: str, block_num: int = 7, dst_w: int = 720):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_image = Image.fromarray(np.uint8(raw_image)).convert('RGB')
        w, h = raw_image.size
        scaling_factor = dst_w / w
        resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
        norm_img = np.float32(resized_img) / 255
        img = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        txt = self.text_processors["eval"](caption)
        txt_tokens = self.model.tokenizer(txt, return_tensors="pt").to(device)
        gradcam, _ = compute_gradcam(self.model, img, txt, txt_tokens, block_num=block_num)
        avg_gradcam_blurred = getAttMap(norm_img, gradcam[0][1], blur=True, overlap=False)
        print(avg_gradcam_blurred.shape, np.min(avg_gradcam_blurred), np.max(avg_gradcam_blurred), np.mean(avg_gradcam_blurred), np.std(avg_gradcam_blurred))
        return avg_gradcam_blurred

    def vali(self, vali_stimuli: List[StimuliStimulus], vali_fixations: FixationTrains, verbose: bool = True):
        self.eval_mode()
        auc_uniform = self.AUC(vali_stimuli, vali_fixations, nonfixations='uniform', verbose=True)
        auc_shuffled = self.AUC(vali_stimuli, vali_fixations, nonfixations='shuffled', verbose=True)
        auc_non_fixations = self.AUC(vali_stimuli, vali_fixations, nonfixations=vali_fixations, verbose=True)
        auc_judd = self.AUC_Judd(vali_stimuli, vali_fixations, verbose=True)
        auc_ss = self.NSS(vali_stimuli, vali_fixations, verbose=True)
        auc_cc = self.CC(vali_stimuli, self, verbose=True)
        auc_sim = self.SIM(vali_stimuli, self, verbose=True)
        dkiv_uniform = self.fixation_based_KL_divergence(vali_stimuli, vali_fixations, nonfixations='uniform')
        dkiv_shuffled = self.fixation_based_KL_divergence(vali_stimuli, vali_fixations, nonfixations='shuffled')
        dkiv_non_fixations = self.fixation_based_KL_divergence(vali_stimuli, vali_fixations, nonfixations=vali_fixations)
        gold_standard = pysaliency.FixationMap(vali_stimuli, vali_fixations, kernel_size=30)
        dkiv_gold = self.image_based_kl_divergence(vali_stimuli, gold_standard)
        logger.info("AUC(uniform):{:02f} | AUC(shuffled):{:02f} | AUC(identical):{:02f} | AUC(Judd):{:02f} | NSS:{:02f} | CC:{:02f} | SIM:{:02f} | Fixation based KL-divergence(uniform):{:02f} | Fixation based KL-divergence(shuffled):{:02f} | Fixation based KL-divergence(identical):{:02f} | Image based KL-divergence:{:02f}".format(
            auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold))
        self.train_mode()
        return auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold

    def train(self, train_stimuli, verbose=True):
        self.train_mode()
        saliency_map = self.saliency_map(train_stimuli)
        return saliency_map

    def test(self, test_stimuli, test_fixations, verbose=True):
        self.eval_mode()
        auc_uniform = self.AUC(test_stimuli, test_fixations, nonfixations='uniform', verbose=True)
        auc_shuffled = self.AUC(test_stimuli, test_fixations, nonfixations='shuffled', verbose=True)
        auc_non_fixations = self.AUC(test_stimuli, test_fixations, nonfixations=test_fixations, verbose=True)
        auc_judd = self.AUC_Judd(test_stimuli, test_fixations, verbose=True)
        auc_ss = self.NSS(test_stimuli, test_fixations, verbose=True)
        auc_cc = self.CC(test_stimuli, self, verbose=True)
        auc_sim = self.SIM(test_stimuli, self, verbose=True)
        dkiv_uniform = self.fixation_based_KL_divergence(test_stimuli, test_fixations, nonfixations='uniform')
        dkiv_shuffled = self.fixation_based_KL_divergence(test_stimuli, test_fixations, nonfixations='shuffled')
        dkiv_non_fixations = self.fixation_based_KL_divergence(test_stimuli, test_fixations, nonfixations=test_fixations)
        gold_standard = pysaliency.FixationMap(test_stimuli, test_fixations, kernel_size=30)
        dkiv_gold = self.image_based_kl_divergence(test_stimuli, gold_standard)
        logger.info("AUC(uniform):{:02f} | AUC(shuffled):{:02f} | AUC(identical):{:02f} | AUC(Judd):{:02f} | NSS:{:02f} | CC:{:02f} | SIM:{:02f} | Fixation based KL-divergence(uniform):{:02f} | Fixation based KL-divergence(shuffled):{:02f} | Fixation based KL-divergence(identical):{:02f} | Image based KL-divergence:{:02f}".format(
            auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold))
        self.train_mode()
        return auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def fit(self, stimuli, fixations, verbose=True):
        """
        Fit the model to the given stimuli and fixations.
        Parameters
        ----------
        stimuli : pysaliency.Stimuli
        fixations : pysaliency.Fixations
        verbose : bool
        """
        dataloader = SaliencyDatasetLoader(stimuli, fixations)
        train_stimuli, train_fixations = dataloader.get_train_data()
        val_stimuli, val_fixations = dataloader.get_val_data()
        test_stimuli, test_fixations = dataloader.get_test_data()

        # Testing
        self.vali(val_stimuli, val_fixations)

        time_now = time.time()
        path = os.path.join(self.args.save_path)

        train_steps = len(train_stimuli)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.train_mode()
            epoch_time = time.time()
            for i, (batch_x, _) in enumerate():
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.train(stimuli, fixations)
                loss = criterion()
                train_loss.append(loss.item())

                if (i + 1) % 400 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold = self.vali(val_stimuli, val_fixations)
            auc_uniform, auc_shuffled, auc_non_fixations, auc_judd, auc_ss, auc_cc, auc_sim, dkiv_uniform, dkiv_shuffled, dkiv_non_fixations, dkiv_gold = self.vali(test_stimuli, test_fixations)

            logger.info("Epoch: {0}, Steps: {1}".format(epoch + 1, train_steps))
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     logger.info("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # region
    def evaluate_saliency_map_for_segments(self, raw_image, model_path, caption, block_num=7, dst_w=720):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", model_path, device=device, is_eval=True)
        raw_image = Image.fromarray(np.uint8(raw_image)).convert('RGB')
        w, h = raw_image.size
        scaling_factor = dst_w / w
        resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
        norm_img = np.float32(resized_img) / 255
        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        txt = text_processors["eval"](caption)
        txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
        gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=block_num)
        avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(avg_gradcam)
        ax.set_yticks([])
        ax.set_xticks([])
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
        return avg_gradcam
 # endregion

# region


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
# endregion

# region


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj == 'type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
# endregion


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune U2M for forecasting')

    # random seed and test random sed
    parser.add_argument('--random_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--test_random_seed', type=int, default=4069, help='random seedm during test time')

    # training or testing
    parser.add_argument('--is_training', action='store_true',
                        help='whether finetuning or perform zero shot imputation', default=False)

    parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--data_name', type=str, default='NIPS_Water', help='data file')
    parser.add_argument('--data_split', type=str, default='0.7, 0.1, 0.2', help='train/val/test split, can be ratio or number')
    parser.add_argument('--checkpoints', type=str, default='checkpoints_forecast/weather/', help='location to store model checkpoints')
    parser.add_argument('--type_name', type=str, default='imputation', help='fixed missing gaps in time series')
    parser.add_argument('--data_format', type=str, default='npy', help='data format', choices=["csv", "npy"])
    parser.add_argument('--slide_step', type=int, default=1, help='sliding steps for the sliding window of train and valid')
    parser.add_argument('--valid_prop', type=float, default=0.2, help='proportion of validation set, for numpy data only')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer initial learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=4, help='experiments times')

    parser.add_argument('--save_path', default="../..exp", type=str, help='')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        logger.warning(
            f"Using multi-gpu support, the device ids are {args.device_ids}")

    print('Args in experiment:')
    # print(args)
    logger.info(args)
    return args


if __name__ == '__main__':

    data_location = 'test_datasets'
    data_location = "../../datasets"
    # mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=data_location)

    text_descriptor = TextDescriptor('../../datasets/sjtuvis/text.xlsx')
    mit_stimuli, mit_fixations = pysaliency.external_datasets.get_sjtu_vis(location=data_location, text_descriptor=text_descriptor)
    # aim = pysaliency.AIM(location='../../models', cache_location=os.path.join('model_caches', 'AIM'))
    # smap = aim.saliency_map(mit_stimuli[10])
    # plt.imshow(-smap)
    # plt.show()
    # print(smap.shape, np.min(-smap), np.max(-smap), np.mean(-smap), np.std(-smap))

    args = parse_args()
    msmm = MySaliencyMapModel(text_descriptor=text_descriptor, args=args)
    msmm.fit(mit_stimuli, mit_fixations, verbose=True)

    cutoff = 10
    short_stimuli = pysaliency.FileStimuli(filenames=mit_stimuli.filenames[:cutoff])
    short_fixations = mit_fixations[mit_fixations.n < cutoff]

    msmm.AUC(short_stimuli, short_fixations, nonfixations='uniform', verbose=True)

    msmm.AUC(short_stimuli, short_fixations, nonfixations='shuffled', verbose=True)

    msmm.AUC(short_stimuli, short_fixations, nonfixations=short_fixations, verbose=True)

    perf = msmm.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations='uniform')
    print('Fixation based KL-divergence wrt. uniform nonfixations: {:.02f}'.format(perf))

    perf = msmm.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations='shuffled')
    print('Fixation based KL-divergence wrt. shuffled nonfixations: {:.02f}'.format(perf))

    perf = msmm.fixation_based_KL_divergence(short_stimuli, short_fixations, nonfixations=short_fixations)
    print('Fixation based KL-divergence wrt. identical nonfixations: {:.02f}'.format(perf))

    gold_standard = pysaliency.FixationMap(short_stimuli, short_fixations, kernel_size=30)
    perf = msmm.image_based_kl_divergence(short_stimuli, gold_standard)
    print("Image based KL-divergence: {} bit".format(perf / np.log(2)))

    gold_standard.image_based_kl_divergence(short_stimuli, gold_standard, minimum_value=1e-20)
