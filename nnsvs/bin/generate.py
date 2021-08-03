# coding: utf-8

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
from tqdm import tqdm
from os.path import basename, splitext, exists, join, abspath
import os
import torch
from torch import nn
from torch.nn import functional as F
from nnmnkwii.datasets import FileSourceDataset

from nnsvs.gen import get_windows
from nnsvs.multistream import multi_stream_mlpg
from nnsvs.bin.train import NpyFileSource
from nnsvs.logger import getLogger
from nnsvs.base import PredictionType
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu, mdn_get_sample

logger = None

use_cuda = torch.cuda.is_available()


@hydra.main(config_path="conf/generate", config_name="config")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if use_cuda else "cpu")
    in_dir = abspath(config.in_dir)
    out_dir = abspath(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_config = OmegaConf.load(abspath(config.model.model_yaml))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    checkpoint = torch.load(abspath(config.model.checkpoint),
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    scaler = joblib.load(abspath(config.out_scaler_path))

    in_feats = FileSourceDataset(NpyFileSource(in_dir))

    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0).to(device)

            if model.prediction_type() == PredictionType.PROBABILISTIC:

                max_mu, max_sigma = model.inference(feats, [feats.shape[1]])

                if np.any(model_config.has_dynamic_features):
                    # Apply denormalization
                    # (B, T, D_out) -> (T, D_out)
                    max_sigma_sq = max_sigma.squeeze(0).cpu().data.numpy() ** 2 * scaler.var_
                    max_mu = scaler.inverse_transform(max_mu.squeeze(0).cpu().data.numpy())
                    # Apply MLPG
                    # (T, D_out) -> (T, static_dim)
                    out = multi_stream_mlpg(max_mu, max_sigma_sq, get_windows(model_config.num_windows),
                                            model_config.stream_sizes, model_config.has_dynamic_features)

                else:
                    # (T, D_out)
                    out = max_mu.squeeze(0).cpu().data.numpy()
                    out = scaler.inverse_transform(out)
            else:
                out = model.inference(feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy()
                out = scaler.inverse_transform(out)

                # Apply MLPG if necessary
                if np.any(model_config.has_dynamic_features):
                    out = multi_stream_mlpg(
                        out, scaler.var_, get_windows(model_config.num_windows),
                        model_config.stream_sizes, model_config.has_dynamic_features)

            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
