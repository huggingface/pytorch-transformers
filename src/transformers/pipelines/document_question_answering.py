# Copyright 2022 The Impira Team and the HuggingFace Team. All rights reserved.
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

import re
from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)
from .base import ChunkPipeline, build_pipeline_init_args
from .question_answering import select_starts_ends


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES

TESSERACT_LOADED = False
if is_pytesseract_available():
    TESSERACT_LOADED = True
    import pytesseract

logger = logging.get_logger(__name__)


# normalize_bbox() and apply_tesseract() are derived from apply_tesseract in models/layoutlmv3/feature_extraction_layoutlmv3.py.
# However, because the pipeline may evolve from what layoutlmv3 currently does, it's copied (vs. imported) to avoid creating an
# unnecessary dependency.
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(image: "Image.Image", lang: Optional[str], tesseract_config: Optional[str]):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    # apply OCR
    data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    image_width, image_height = image.size

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    if len(words) != len(normalized_boxes):
        raise ValueError("Not as many words as there are bounding boxes")

    return words, normalized_boxes


class ModelType(ExplicitEnum):
    LayoutLM = "layoutlm"
    LayoutLMv2andv3 = "layoutlmv2andv3"
    VisionEncoderDecoder = "vision_encoder_decoder"


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True, has_tokenizer=True))
class DocumentQuestionAnsweringPipeline(ChunkPipeline):
    # TODO: Update task_summary docs to include an example with document QA and then update the first sentence
    """
    Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. The inputs/outputs are
    similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR'd
    words/boxes) as input instead of text context.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> document_qa = pipeline(model="impira/layoutlm-document-qa")
    >>> document_qa(
    ...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    ...     question="What is the invoice number?",
    ... )
    [{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This document question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"document-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tokenizer is not None and not self.tokenizer.__class__.__name__.endswith("Fast"):
            raise ValueError(
                "`DocumentQuestionAnsweringPipeline` requires a fast tokenizer, but a slow tokenizer "
                f"(`{self.tokenizer.__class__.__name__}`) is provided."
            )

        if self.model.config.__class__.__name__ == "VisionEncoderDecoderConfig":
            self.model_type = ModelType.VisionEncoderDecoder
            if self.model.config.encoder.model_type != "donut-swin":
                raise ValueError("Currently, the only supported VisionEncoderDecoder model is Donut")
        else:
            self.check_model_type(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES)
            if self.model.config.__class__.__name__ == "LayoutLMConfig":
                self.model_type = ModelType.LayoutLM
            else:
                self.model_type = ModelType.LayoutLMv2andv3

    def _sanitize_parameters(
        self,
        padding=None,
        doc_stride=None,
        max_question_len=None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        max_answer_len=None,
        max_seq_len=None,
        top_k=None,
        handle_impossible_answer=None,
        timeout=None,
        **kwargs,
    ):
        preprocess_params, postprocess_params = {}, {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len
        if lang is not None:
            preprocess_params["lang"] = lang
        if tesseract_config is not None:
            preprocess_params["tesseract_config"] = tesseract_config
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            if max_answer_len < 1:
                raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer

        forward_params = {}
        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, postprocess_params

    def __call__(
        self,
        image: Union["Image.Image", str],
        question: Optional[str] = None,
        word_boxes: Tuple[str, List[float]] = None,
        **kwargs,
    ):
        """
        Answer the question(s) given as inputs by using the document(s). A document is defined as an image and an
        optional list of (word, box) tuples which represent the text in the document. If the `word_boxes` are not
        provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for
        LayoutLM-like models which require them as input. For Donut, no OCR is run.

        You can invoke the pipeline several ways:

        - `pipeline(image=image, question=question)`
        - `pipeline(image=image, question=question, word_boxes=word_boxes)`
        - `pipeline([{"image": image, "question": question}])`
        - `pipeline([{"image": image, "question": question, "word_boxes": word_boxes}])`

        Args:
            image (`str` or `PIL.Image`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. If given a single image, it can be
                broadcasted to multiple questions.
            question (`str`):
                A question to ask of the document.
            word_boxes (`List[str, Tuple[float, float, float, float]]`, *optional*):
                A list of words and bounding boxes (normalized 0->1000). If you provide this optional input, then the
                pipeline will use these words and boxes instead of running OCR on the image to derive them for models
                that need them (e.g. LayoutLM). This allows you to reuse OCR'd results across many invocations of the
                pipeline without having to re-run it each time.
            top_k (`int`, *optional*, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                top_k answers if there are not enough options available within the context.
            doc_stride (`int`, *optional*, defaults to 128):
                If the words in the document are too long to fit with the question for the model, it will be split in
                several chunks with some overlap. This argument controls the size of that overlap.
            max_answer_len (`int`, *optional*, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (`int`, *optional*, defaults to 384):
                The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
                model. The context will be split in several chunks (using `doc_stride` as overlap) if needed.
            max_question_len (`int`, *optional*, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (`bool`, *optional*, defaults to `False`):
                Whether or not we accept impossible as an answer.
            lang (`str`, *optional*):
                Language to use while running OCR. Defaults to english.
            tesseract_config (`str`, *optional*):
                Additional flags to pass to tesseract while running OCR.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:

            - **score** (`float`) -- The probability associated to the answer.
            - **start** (`int`) -- The start word index of the answer (in the OCR'd version of the input or provided
              `word_boxes`).
            - **end** (`int`) -- The end word index of the answer (in the OCR'd version of the input or provided
              `word_boxes`).
            - **answer** (`str`) -- The answer to the question.
            - **words** (`list[int]`) -- The index of each word/box pair that is in the answer
        """
        if isinstance(question, str):
            inputs = {"question": question, "image": image}
            if word_boxes is not None:
                inputs["word_boxes"] = word_boxes
        else:
            inputs = image
        return super().__call__(inputs, **kwargs)

    def preprocess(
        self,
        input,
        padding="do_not_pad",
        doc_stride=None,
        max_seq_len=None,
        word_boxes: Tuple[str, List[float]] = None,
        lang=None,
        tesseract_config="",
        timeout=None,
    ):
        # NOTE: This code mirrors the code in question answering and will be implemented in a follow up PR
        # to support documents with enough tokens that overflow the model's window
        if max_seq_len is None:
            max_seq_len = self.tokenizer.model_max_length

        if doc_stride is None:
            doc_stride = min(max_seq_len // 2, 256)

        image = None
        image_features = {}
        if input.get("image", None) is not None:
            image = load_image(input["image"], timeout=timeout)
            if self.image_processor is not None:
                image_inputs = self.image_processor(images=image, return_tensors=self.framework)
                if self.framework == "pt":
                    image_inputs = image_inputs.to(self.torch_dtype)
                image_features.update(image_inputs)
            elif self.feature_extractor is not None:
                image_features.update(self.feature_extractor(images=image, return_tensors=self.framework))
            elif self.model_type == ModelType.VisionEncoderDecoder:
                raise ValueError("If you are using a VisionEncoderDecoderModel, you must provide a feature extractor")

        words, boxes = None, None
        if not self.model_type == ModelType.VisionEncoderDecoder:
            if "word_boxes" in input:
                words = [x[0] for x in input["word_boxes"]]
                boxes = [x[1] for x in input["word_boxes"]]
            elif "words" in image_features and "boxes" in image_features:
                words = image_features.pop("words")[0]
                boxes = image_features.pop("boxes")[0]
            elif image is not None:
                if not TESSERACT_LOADED:
                    raise ValueError(
                        "If you provide an image without word_boxes, then the pipeline will run OCR using Tesseract,"
                        " but pytesseract is not available"
                    )
                if TESSERACT_LOADED:
                    words, boxes = apply_tesseract(image, lang=lang, tesseract_config=tesseract_config)
            else:
                raise ValueError(
                    "You must provide an image or word_boxes. If you provide an image, the pipeline will automatically"
                    " run OCR to derive words and boxes"
                )

        if self.tokenizer.padding_side != "right":
            raise ValueError(
                "Document question answering only supports tokenizers whose padding side is 'right', not"
                f" {self.tokenizer.padding_side}"
            )

        if self.model_type == ModelType.VisionEncoderDecoder:
            task_prompt = f'<s_docvqa><s_question>{input["question"]}</s_question><s_answer>'
            # Adapted from https://huggingface.co/spaces/nielsr/donut-docvqa/blob/main/app.py
            encoding = {
                "inputs": image_features["pixel_values"],
                "decoder_input_ids": self.tokenizer(
                    task_prompt, add_special_tokens=False, return_tensors=self.framework
                ).input_ids,
                "return_dict_in_generate": True,
            }
            yield {
                **encoding,
                "p_mask": None,
                "word_ids": None,
                "words": None,
                "output_attentions": True,
                "is_last": True,
            }
        else:
            tokenizer_kwargs = {}
            if self.model_type == ModelType.LayoutLM:
                tokenizer_kwargs["text"] = input["question"].split()
                tokenizer_kwargs["text_pair"] = words
                tokenizer_kwargs["is_split_into_words"] = True
            else:
                tokenizer_kwargs["text"] = [input["question"]]
                tokenizer_kwargs["text_pair"] = [words]
                tokenizer_kwargs["boxes"] = [boxes]

            encoding = self.tokenizer(
                padding=padding,
                max_length=max_seq_len,
                stride=doc_stride,
                return_token_type_ids=True,
                truncation="only_second",
                return_overflowing_tokens=True,
                **tokenizer_kwargs,
            )
            # TODO: check why slower `LayoutLMTokenizer` and `LayoutLMv2Tokenizer` don't have this key in outputs
            # FIXME: ydshieh and/or Narsil
            encoding.pop("overflow_to_sample_mapping", None)  # We do not use this

            num_spans = len(encoding["input_ids"])

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
            # This logic mirrors the logic in the question_answering pipeline
            p_mask = np.array([[tok != 1 for tok in encoding.sequence_ids(span_id)] for span_id in range(num_spans)])
            for span_idx in range(num_spans):
                if self.framework == "pt":
                    span_encoding = {k: torch.tensor(v[span_idx : span_idx + 1]) for (k, v) in encoding.items()}
                    if "pixel_values" in image_features:
                        span_encoding["image"] = image_features["pixel_values"]
                else:
                    raise ValueError("Unsupported: Tensorflow preprocessing for DocumentQuestionAnsweringPipeline")

                input_ids_span_idx = encoding["input_ids"][span_idx]
                # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
                if self.tokenizer.cls_token_id is not None:
                    cls_indices = np.nonzero(np.array(input_ids_span_idx) == self.tokenizer.cls_token_id)[0]
                    for cls_index in cls_indices:
                        p_mask[span_idx][cls_index] = 0

                # For each span, place a bounding box [0,0,0,0] for question and CLS tokens, [1000,1000,1000,1000]
                # for SEP tokens, and the word's bounding box for words in the original document.
                if "boxes" not in tokenizer_kwargs:
                    bbox = []
                    for input_id, sequence_id, word_id in zip(
                        encoding.input_ids[span_idx],
                        encoding.sequence_ids(span_idx),
                        encoding.word_ids(span_idx),
                    ):
                        if sequence_id == 1:
                            bbox.append(boxes[word_id])
                        elif input_id == self.tokenizer.sep_token_id:
                            bbox.append([1000] * 4)
                        else:
                            bbox.append([0] * 4)

                    if self.framework == "pt":
                        span_encoding["bbox"] = torch.tensor(bbox).unsqueeze(0)
                    elif self.framework == "tf":
                        raise ValueError("Unsupported: Tensorflow preprocessing for DocumentQuestionAnsweringPipeline")
                yield {
                    **span_encoding,
                    "p_mask": p_mask[span_idx],
                    "word_ids": encoding.word_ids(span_idx),
                    "words": words,
                    "is_last": span_idx == num_spans - 1,
                }

    def _forward(self, model_inputs, **generate_kwargs):
        p_mask = model_inputs.pop("p_mask", None)
        word_ids = model_inputs.pop("word_ids", None)
        words = model_inputs.pop("words", None)
        is_last = model_inputs.pop("is_last", False)

        if self.model_type == ModelType.VisionEncoderDecoder:
            # User-defined `generation_config` passed to the pipeline call take precedence
            if "generation_config" not in generate_kwargs:
                generate_kwargs["generation_config"] = self.generation_config

            model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        else:
            model_outputs = self.model(**model_inputs)

        model_outputs = dict(model_outputs.items())
        model_outputs["p_mask"] = p_mask
        model_outputs["word_ids"] = word_ids
        model_outputs["words"] = words
        model_outputs["attention_mask"] = model_inputs.get("attention_mask", None)
        model_outputs["is_last"] = is_last
        return model_outputs

    def postprocess(self, model_outputs, top_k=1, **kwargs):
        if self.model_type == ModelType.VisionEncoderDecoder:
            answers = [self.postprocess_encoder_decoder_single(o) for o in model_outputs]
        else:
            answers = self.postprocess_extractive_qa(model_outputs, top_k=top_k, **kwargs)

        answers = sorted(answers, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        return answers

    def postprocess_encoder_decoder_single(self, model_outputs, **kwargs):
        sequence = self.tokenizer.batch_decode(model_outputs["sequences"])[0]

        # TODO: A lot of this logic is specific to Donut and should probably be handled in the tokenizer
        # (see https://github.com/huggingface/transformers/pull/18414/files#r961747408 for more context).
        sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        ret = {
            "answer": None,
        }

        answer = re.search(r"<s_answer>(.*)</s_answer>", sequence)
        if answer is not None:
            ret["answer"] = answer.group(1).strip()
        return ret

    def postprocess_extractive_qa(
        self, model_outputs, top_k=1, handle_impossible_answer=False, max_answer_len=15, **kwargs
    ):
        min_null_score = 1000000  # large and positive
        answers = []
        for output in model_outputs:
            words = output["words"]

            starts, ends, scores, min_null_score = select_starts_ends(
                start=output["start_logits"],
                end=output["end_logits"],
                p_mask=output["p_mask"],
                attention_mask=output["attention_mask"].numpy()
                if output.get("attention_mask", None) is not None
                else None,
                min_null_score=min_null_score,
                top_k=top_k,
                handle_impossible_answer=handle_impossible_answer,
                max_answer_len=max_answer_len,
            )
            word_ids = output["word_ids"]
            for start, end, score in zip(starts, ends, scores):
                word_start, word_end = word_ids[start], word_ids[end]
                if word_start is not None and word_end is not None:
                    answers.append(
                        {
                            "score": float(score),
                            "answer": " ".join(words[word_start : word_end + 1]),
                            "start": word_start,
                            "end": word_end,
                        }
                    )

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "answer": "", "start": 0, "end": 0})

        return answers
