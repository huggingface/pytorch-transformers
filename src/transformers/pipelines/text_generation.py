import enum
import itertools
import types
from typing import Dict

from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from .pt_utils import KeyDataset

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any `ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt. When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Examples:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="openai-community/gpt2")
    >>> generator("I can't believe you did such a ", do_sample=False)
    [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

    >>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
    >>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
    ```

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
    >>> # Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
    >>> generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
    [{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This language generation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective. See the list of available [text completion models](https://huggingface.co/models?filter=text-generation)
    and the list of [conversational models](https://huggingface.co/models?other=conversational)
    on [huggingface.co/models].
    """

    # Prefix text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
    # in https://github.com/rusiaaman/XLNet-gen#methodology
    # and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e

    XL_PREFIX = """
    In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The
    voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western
    Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision
    and denounces one of the men as a horse thief. Although his father initially slaps him for making such an
    accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop,
    begging for his blessing. <eod> </s> <eos>
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        )
        if "prefix" not in self._preprocess_params:
            # This is very specific. The logic is quite complex and needs to be done
            # as a "default".
            # It also defines both some preprocess_kwargs and generate_kwargs
            # which is why we cannot put them in their respective methods.
            prefix = None
            if self.prefix is not None:
                prefix = self.prefix
            if prefix is None and self.model.__class__.__name__ in [
                "XLNetLMHeadModel",
                "TransfoXLLMHeadModel",
                "TFXLNetLMHeadModel",
                "TFTransfoXLLMHeadModel",
            ]:
                # For XLNet and TransformerXL we add an article to the prompt to give more state to the model.
                prefix = self.XL_PREFIX
            if prefix is not None:
                # Recalculate some generate_kwargs linked to prefix.
                preprocess_params, forward_params, _ = self._sanitize_parameters(prefix=prefix, **self._forward_params)
                self._preprocess_params = {**self._preprocess_params, **preprocess_params}
                self._forward_params = {**self._forward_params, **forward_params}

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        truncation=None,
        max_length=None,
        continue_final_message=None,
        **generate_kwargs,
    ):
        preprocess_params = {}

        add_special_tokens = False
        if "add_special_tokens" in generate_kwargs:
            add_special_tokens = preprocess_params["add_special_tokens"] = generate_kwargs.pop("add_special_tokens")

        if "padding" in generate_kwargs:
            preprocess_params["padding"] = generate_kwargs.pop("padding")

        if truncation is not None:
            preprocess_params["truncation"] = truncation

        if max_length is not None:
            preprocess_params["max_length"] = max_length
            generate_kwargs["max_length"] = max_length

        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=add_special_tokens, return_tensors=self.framework
            )
            generate_kwargs["prefix_length"] = prefix_inputs["input_ids"].shape[-1]

        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        if continue_final_message is not None:
            preprocess_params["continue_final_message"] = continue_final_message

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_full_text`")
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError("`return_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        if continue_final_message is not None:
            postprocess_params["continue_final_message"] = continue_final_message

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            generate_kwargs["eos_token_id"] = stop_sequence_ids

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, postprocess_params

    # overriding _parse_and_tokenize to allow for unusual language-modeling tokenizer arguments
    def _parse_and_tokenize(self, *args, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            kwargs.update({"add_space_before_punct_symbol": True})

        return super()._parse_and_tokenize(*args, **kwargs)

    def __call__(self, text_inputs, **kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            text_inputs (`str`, `List[str]`, List[Dict[str, str]], or `List[List[Dict[str, str]]]`):
                One or several prompts (or one list of prompts) to complete. If strings or a list of string are
                passed, this pipeline will continue each prompt. Alternatively, a "chat", in the form of a list
                of dicts with "role" and "content" keys, can be passed, or a list of such chats. When chats are passed,
                the model's chat template will be used to format them before passing them to the model.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Returns the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*):
                Returns the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
                specified at the same time as `return_text`.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            continue_final_message( `bool`, *optional*): This indicates that you want the model to continue the
                last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
                By default this is `True` when the final message in the input chat has the `assistant` role and
                `False` otherwise, but you can manually override that behaviour by setting this flag.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to adress this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)
            generate_kwargs (`dict`, *optional*):
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./text_generation)).

        Return:
            A list or a list of lists of `dict`: Returns one of the following dictionaries (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """
        if isinstance(
            text_inputs,
            (list, tuple, types.GeneratorType, KeyDataset)
            if is_torch_available()
            else (list, tuple, types.GeneratorType),
        ):
            if isinstance(text_inputs, types.GeneratorType):
                text_inputs, _ = itertools.tee(text_inputs)
                text_inputs, first_item = (x for x in text_inputs), next(_)
            else:
                first_item = text_inputs[0]
            if isinstance(first_item, (list, tuple, dict)):
                # We have one or more prompts in list-of-dicts format, so this is chat mode
                if isinstance(first_item, dict):
                    return super().__call__(Chat(text_inputs), **kwargs)
                else:
                    chats = (Chat(chat) for chat in text_inputs)  # 🐈 🐈 🐈
                    if isinstance(text_inputs, types.GeneratorType):
                        return super().__call__(chats, **kwargs)
                    else:
                        return super().__call__(list(chats), **kwargs)
        return super().__call__(text_inputs, **kwargs)

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        add_special_tokens=None,
        truncation=None,
        padding=None,
        max_length=None,
        continue_final_message=None,
        **generate_kwargs,
    ):
        # Only set non-None tokenizer kwargs, so as to rely on the tokenizer's defaults
        tokenizer_kwargs = {
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
            "padding": padding,
            "max_length": max_length,
        }
        tokenizer_kwargs = {key: value for key, value in tokenizer_kwargs.items() if value is not None}

        if isinstance(prompt_text, Chat):
            tokenizer_kwargs.pop("add_special_tokens", None)  # ignore add_special_tokens on chats
            # If the user passes a chat that ends in an assistant message, we treat it as a prefill by default
            # because very few models support multiple separate, consecutive assistant messages
            if continue_final_message is None:
                continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
            inputs = self.tokenizer.apply_chat_template(
                prompt_text.messages,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
                return_dict=True,
                return_tensors=self.framework,
                **tokenizer_kwargs,
            )
        else:
            inputs = self.tokenizer(prefix + prompt_text, return_tensors=self.framework, **tokenizer_kwargs)

        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = generate_kwargs.get("max_length", self.generation_config.max_length) - cur_len
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the"
                        " models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, -keep_length:]

        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.generation_config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}

    def postprocess(
        self,
        model_outputs,
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
        continue_final_message=None,
    ):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": sequence}
            elif return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
                # Decode text
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )

                # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    )

                all_text = text[prompt_length:]
                if return_type == ReturnType.FULL_TEXT:
                    if isinstance(prompt_text, str):
                        all_text = prompt_text + all_text
                    elif isinstance(prompt_text, Chat):
                        if continue_final_message is None:
                            # If the user passes a chat ending in an assistant message, we treat it as a prefill by
                            # default because very few models support multiple separate, consecutive assistant messages
                            continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
                        if continue_final_message:
                            # With assistant prefill, concat onto the end of the last message
                            all_text = list(prompt_text.messages)[:-1] + [
                                {
                                    "role": prompt_text.messages[-1]["role"],
                                    "content": prompt_text.messages[-1]["content"] + all_text,
                                }
                            ]
                        else:
                            # When we're not starting from a prefill, the output is a new assistant message
                            all_text = list(prompt_text.messages) + [{"role": "assistant", "content": all_text}]
                record = {"generated_text": all_text}
            records.append(record)

        return records
