from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)


__import_structure = {
    "configuration_sam_hq": [
        "SamHQConfig",
        "SamHQMaskDecoderConfig",
        "SamHQPromptEncoderConfig",
        "SamHQVisionConfig",
    ],
    "processing_samhq": ["SamHQProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    __import_structure["modeling_sam_hq"] = [
        "SamHQModel",
        "SamHQPreTrainedModel",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    __import_structure["image_processing_sam"] = ["SamImageProcessor"]


if TYPE_CHECKING:
    from .configuration_sam_hq import (
        SamHQConfig,
        SamHQMaskDecoderConfig,
        SamHQPromptEncoderConfig,
        SamHQVisionConfig,
    )
    from .processing_samhq import SamHQProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_sam_hq import SamHQModel, SamHQPreTrainedModel

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from transformers.models.sam.image_processing_sam import SamImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], __import_structure, module_spec=__spec__)
