# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for ProPainter.
"""

import sys
from typing import Dict, Optional


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from ...feature_extraction_utils import BatchFeature
from ...image_utils import VideoInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    VideosKwargs,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class ProPainterVideosKwargs(VideosKwargs, total=False):
    video_painting_mode: str
    scale_size: Optional[tuple[float, float]]
    mask_dilation: int


class ProPainterProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    video_kwargs: ProPainterVideosKwargs
    _defaults = {
        "video_kwargs": {
            "video_painting_mode": "video_inpainting",
            "mask_dilation": 4,
        },
    }


class ProPainterProcessor(ProcessorMixin):
    r"""
    Constructs a ProPainter processor which wraps and abstract a ProPainter video processor into a single processor.

    [`ProPainterProcessor`] offers all the functionalities of [`ProPainterVideoProcessor`]. See the [`~ProPainterVideoProcessor.__call__`],
    for more information.

    Args:
        video_processor ([`ProPainterVideoProcessor`], *optional*):
            The video processor is a required input.
    """

    attributes = ["video_processor"]
    video_processor_class = "ProPainterVideoProcessor"

    def __init__(
        self,
        video_processor=None,
        **kwargs,
    ):
        super().__init__(video_processor)

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        **kwargs,
    ) -> Dict[str, Dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality in output_kwargs:
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if kwarg_value != "__empty__":
                    output_kwargs[modality][modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key in kwargs:
                if key not in used_keys:
                    output_kwargs["common_kwargs"][key] = kwargs[key]

        # all modality-specific kwargs are updated with common kwargs
        for modality in output_kwargs:
            output_kwargs[modality].update(output_kwargs["common_kwargs"])
        return output_kwargs

    def __call__(
        self,
        videos: VideoInput = None,
        masks: VideoInput = None,
        **kwargs: Unpack[ProPainterProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several video(s) and their respective masks for all the frames. To prepare the video(s)
        and masks, this method forwards the `videos`, `masks` and `kwrags` arguments to ProPainterProcessor's
        [`~ProPainterProcessor.__call__`] if `videos` and `masks` are not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The video or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
            masks (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The masks(for all frames of a single video) or batch of masks to be prepared. Each set of masks for a single video
                can be a 4D NumPy array or PyTorch

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **flow_masks** -- Pixel values of flow masks for all frames of a video input to be fed to a model. Returned when `masks` is not `None`.
            - **masks_dilated** -- Pixel values of dilated masks for all frames of a video input to be fed to a model. Returned when `masks` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            ProPainterProcessorKwargs,
            **kwargs,
        )

        video_inputs = {}

        if videos is not None and masks is not None:
            video_inputs = self.video_processor(videos, masks=masks, **output_kwargs["videos_kwargs"])

        return BatchFeature(data={**video_inputs})

    @property
    # Adapted from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        video_processor_input_names = self.video_processor.model_input_names
        return list(dict.fromkeys(video_processor_input_names))
