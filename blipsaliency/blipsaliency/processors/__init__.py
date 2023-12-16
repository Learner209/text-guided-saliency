"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from blipsaliency.processors.base_processor import BaseProcessor
from blipsaliency.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
)
from blipsaliency.common.registry import registry

__all__ = [
    "BaseProcessor",
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor"
]


def load_processor(name, cfg=None):
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
