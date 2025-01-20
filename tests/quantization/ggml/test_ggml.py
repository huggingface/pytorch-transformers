# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

from transformers import AddedToken, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import (
    require_gguf,
    require_read_token,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_gguf
@require_torch_gpu
@slow
class GgufIntegrationTests(unittest.TestCase):
    original_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    imatrix_model_id = "duyntnet/TinyLlama-1.1B-Chat-v1.0-imatrix-GGUF"
    mistral_model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    qwen2_model_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
    qwen2moe_model_id = "gdax/Qwen1.5-MoE-A2.7B_gguf"
    qwen2moe_original_model_id = "Qwen/Qwen1.5-MoE-A2.7B"
    llama3_model_id = "NousResearch/Meta-Llama-3-8B-GGUF"
    tinyllama_model_id = "PenutChen/TinyLlama-1.1B-Chat-v1.0-GGUF"
    phi3_model_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    bloom_model_id = "afrideva/bloom-560m-GGUF"
    original_bloom_model_id = "bigscience/bloom-560m"
    falcon7b_model_id_q2 = "xaviviro/falcon-7b-quantized-gguf"
    falcon7b_model_id_fp16 = "medmekk/falcon-7b-gguf"
    falcon40b_model_id = "maddes8cht/tiiuae-falcon-40b-gguf"
    original_flacon7b_model_id = "tiiuae/falcon-7b"
    t5_model_id = "repetitio/flan-t5-small"
    original_t5_model_id = "google/flan-t5-small"
    stablelm_model_id = "afrideva/stablelm-3b-4e1t-GGUF"
    stablelm2_model_id = "afrideva/stablelm-2-1_6b-GGUF"
    original_stablelm2_model_id = "stabilityai/stablelm-2-1_6b"
    gpt2_model_id = "mradermacher/gpt2-GGUF"
    gpt2_original_model_id = "openai-community/gpt2"
    gpt2_xl_model_id = "RichardErkhov/openai-community_-_gpt2-xl-gguf"
    starcoder2_model_id = "QuantFactory/starcoder2-3b-GGUF"
    starcoder2_fp16_model_id = "brittlewis12/starcoder2-3b-GGUF"
    starcoder2_original_model_id = "bigcode/starcoder2-3b"
    mamba_original_model_id = "state-spaces/mamba-2.8b-hf"
    mamba_model_id = "jpodivin/mamba-2.8b-hf-GGUF"
    nemotron_original_model_id = "nvidia/Nemotron-Mini-4B-Instruct"
    nemotron_model_id = "bartowski/Nemotron-Mini-4B-Instruct-GGUF"
    original_gemma2_model_id = "google/gemma-2-2b-it"
    gemma2_model_id = "bartowski/gemma-2-2b-it-GGUF"

    # standard quants
    q4_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    q5_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q5_0.gguf"
    q8_0_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    # k-quants
    q2_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    q3_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf"
    q4_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    q5_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    q6_k_gguf_model_id = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
    q4_k_m_stablelm_model_id = "stablelm-3b-4e1t.q4_k_m.gguf"
    # imatrix
    iq1_m_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ1_M.gguf"
    iq1_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ1_S.gguf"
    iq2_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_S.gguf"
    iq2_xs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_XS.gguf"
    iq2_xxs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ2_XXS.gguf"
    iq3_s_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ3_S.gguf"
    iq3_xxs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ3_XXS.gguf"
    iq4_xs_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ4_XS.gguf"
    iq4_nl_gguf_model_id = "TinyLlama-1.1B-Chat-v1.0-IQ4_NL.gguf"

    q4_0_phi3_model_id = "Phi-3-mini-4k-instruct-q4.gguf"
    q4_0_mistral_model_id = "mistral-7b-instruct-v0.2.Q4_0.gguf"
    q4_0_qwen2_model_id = "qwen1_5-0_5b-chat-q4_0.gguf"
    q8_qwen2moe_model_id = "Qwen1.5-MoE-A2.7B_Q8_0.gguf"
    q4_llama3_model_id = "Meta-Llama-3-8B-Q4_K_M.gguf"
    fp16_bloom_model_id = "bloom-560m.fp16.gguf"
    fp16_stablelm2_model_id = "stablelm-2-1_6b.fp16.gguf"
    q8_bloom_model_id = "bloom-560m.q8_0.gguf"
    f16_tinyllama_model_id = "TinyLlama-1.1B-Chat-v1.0.FP16.gguf"
    q2_k_falcon7b_model_id = "falcon-7b-q2_k.gguf"
    fp16_falcon7b_model_id = "falcon-7b-fp16.gguf"
    q2_k_falcon40b_model_id = "tiiuae-falcon-40b-Q2_K.gguf"
    fp16_t5_model_id = "flan-t5-small-f16.gguf"
    q8_0_t5_model_id = "flan-t5-small-q8_0.gguf"
    fp16_qwen2moe_model_id = "Qwen1.5-MoE-A2.7B.gguf"
    fp16_gpt2_model_id = "gpt2.f16.gguf"
    q8_gpt2_model_id = "gpt2.Q8_0.gguf"
    q6_k_gpt2_xl_model_id = "gpt2-xl.Q6_K.gguf"
    q6_k_starcoder2_model_id = "starcoder2-3b.Q6_K.gguf"
    fp16_starcoder2_gguf_model_id = "starcoder2-3b.fp16.gguf"
    q6_k_mamba_model_id = "ggml-model-Q6_K.gguf"
    fp16_mamba_model_id = "ggml-model-f16.gguf"
    q6_k_nemotron_model_id = "Nemotron-Mini-4B-Instruct-Q6_K.gguf"
    fp16_nemotron_model_id = "Nemotron-Mini-4B-Instruct-f16.gguf"
    q3_k_gemma2_model_id = "gemma-2-2b-it-Q3_K_L.gguf"
    q8_0_gemma2_model_id = "gemma-2-2b-it-Q8_0.gguf"
    fp32_gemma2_model_id = "gemma-2-2b-it-f32.gguf"

    example_text = "Hello"

    def test_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q2_k_serialization(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q2_k_gguf_model_id).to(torch_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            tokenizer.save_pretrained(tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(tmpdirname).to(torch_device)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)

            text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
            out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n[10:0"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q3_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q3_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q3_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n```\n<|user"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q5_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q5_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q5_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Use a library"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q5_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q5_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q5_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q4_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q4_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q4_k_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q4_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q4_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Python:\n"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q6_k_fp16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q6_k_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, gguf_file=self.q6_k_gguf_model_id, torch_dtype=torch.float16
        ).to(torch_device)

        self.assertTrue(model.lm_head.weight.dtype == torch.float16)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\nStep 3: Add"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Use a library"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq1_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a friend of mine, I"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq1_m(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_m_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq1_m_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in purching a copy of"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello World!\n\n```\n<|user|"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_xs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello World!\n\n```\n<|user|"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq2_xxs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xxs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq2_xxs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a software engineer. I'"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq3_s(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_s_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_s_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Python:\n"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq3_xxs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_xxs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq3_xxs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in your product. Can you"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq4_xs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_xs_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_xs_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, world!\n\n5. Using a loop"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_iq4_nl(self):
        tokenizer = AutoTokenizer.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_nl_gguf_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.imatrix_model_id, gguf_file=self.iq4_nl_gguf_model_id).to(
            torch_device
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, world!\n\n5. Using a loop"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_f16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_model_id, gguf_file=self.f16_tinyllama_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.tinyllama_model_id, gguf_file=self.f16_tinyllama_model_id
        ).to(torch_device)

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, World!\n\n5. Node.js"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_mistral_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.mistral_model_id, gguf_file=self.q4_0_mistral_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.mistral_model_id,
            gguf_file=self.q4_0_mistral_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello,\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen2_model_id, gguf_file=self.q4_0_qwen2_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen2_model_id,
            gguf_file=self.q4_0_qwen2_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello.jsoup\n\nI am a beginner"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2moe_q8(self):
        tokenizer = AutoTokenizer.from_pretrained(self.qwen2moe_model_id, gguf_file=self.q8_qwen2moe_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_model_id,
            gguf_file=self.q8_qwen2moe_model_id,
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a 20 year old male"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_qwen2moe_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_model_id,
            gguf_file=self.fp16_qwen2moe_model_id,
            torch_dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.qwen2moe_original_model_id,
            torch_dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])

    def test_phi3_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.phi3_model_id, gguf_file=self.q4_0_phi3_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.phi3_model_id, gguf_file=self.q4_0_phi3_model_id, device_map="auto", torch_dtype=torch.float16
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I've been reading about the impact of"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_llama3_q4_0_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llama3_model_id, gguf_file=self.q4_llama3_model_id)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save_pretrained(tmpdirname)
            tokenizer = AutoTokenizer.from_pretrained(tmpdirname)
            special_sentence = "สวัสดี"
            predicted_text = tokenizer.decode(tokenizer.encode(special_sentence, return_tensors="pt")[0])
            self.assertEqual(predicted_text, "<|begin_of_text|>" + special_sentence)

    def test_llama3_q4_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llama3_model_id, gguf_file=self.q4_llama3_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.llama3_model_id,
            gguf_file=self.q4_llama3_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am interested in [The Park]\nThe"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_fp16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.bloom_model_id, gguf_file=self.fp16_bloom_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.fp16_bloom_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I just want to say that I am very"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.bloom_model_id, gguf_file=self.q8_bloom_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.q8_bloom_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I just want to say that I am just"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_bloom_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.bloom_model_id,
            gguf_file=self.fp16_bloom_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_bloom_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for (quantized_name, quantized_param), (original_name, original_param) in zip(
            quantized_state_dict.items(), original_state_dict.items()
        ):
            if (
                "self_attention.query_key_value" in quantized_name
                and "self_attention.query_key_value" in original_name
            ):
                self.assertTrue(quantized_param.shape == original_param.shape)
                torch.testing.assert_close(quantized_param, original_param)

    def test_t5_f16(self):
        tokenizer = AutoTokenizer.from_pretrained(self.t5_model_id, gguf_file=self.fp16_t5_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id, gguf_file=self.fp16_t5_model_id, device_map="auto", torch_dtype=torch.float16
        )

        T5_EXAMPLE_TEXT = "translate English to German: How old are you?"

        text = tokenizer(T5_EXAMPLE_TEXT, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Wie ich er?"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_t5_q8_0(self):
        tokenizer = AutoTokenizer.from_pretrained(self.t5_model_id, gguf_file=self.q8_0_t5_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id, gguf_file=self.q8_0_t5_model_id, device_map="auto", torch_dtype=torch.float16
        )

        T5_EXAMPLE_TEXT = "translate English to German: How old are you?"

        text = tokenizer(T5_EXAMPLE_TEXT, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Wie ich er?"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_t5_weights_conversion_fp16(self):
        quantized_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.t5_model_id,
            gguf_file=self.fp16_t5_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        original_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.original_t5_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for (quantized_name, quantized_param), (original_name, original_param) in zip(
            quantized_state_dict.items(), original_state_dict.items()
        ):
            self.assertTrue(quantized_param.shape == original_param.shape)
            torch.testing.assert_close(quantized_param, original_param, rtol=5e-04, atol=5e-04)

    def test_gpt2_q8(self):
        tokenizer = AutoTokenizer.from_pretrained(self.gpt2_model_id, gguf_file=self.q8_gpt2_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_model_id,
            gguf_file=self.q8_gpt2_model_id,
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm sorry. I'm sorry. I"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gpt2_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_model_id,
            gguf_file=self.fp16_gpt2_model_id,
            torch_dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_original_model_id,
            torch_dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_gpt2_xl_Q6_K(self):
        tokenizer = AutoTokenizer.from_pretrained(self.gpt2_xl_model_id, gguf_file=self.q6_k_gpt2_xl_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.gpt2_xl_model_id,
            gguf_file=self.q6_k_gpt2_xl_model_id,
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I'm a newbie to the world of"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @unittest.skip(reason="Heavy memory")
    def test_falcon40b_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.falcon40b_model_id, gguf_file=self.q2_k_falcon40b_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.falcon40b_model_id,
            gguf_file=self.q2_k_falcon40b_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello All,\nI am new to this forum."
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_falcon7b_q2_k(self):
        tokenizer = AutoTokenizer.from_pretrained(self.falcon7b_model_id_q2, gguf_file=self.q2_k_falcon7b_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.falcon7b_model_id_q2,
            gguf_file=self.q2_k_falcon7b_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"].to(torch_device)
        out = model.generate(text, max_new_tokens=16)

        EXPECTED_TEXT = 'Hello,\nI am trying to use the "get_post_meta"'
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_falcon7b_weights_conversion_fp16(self):
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.falcon7b_model_id_fp16,
            gguf_file=self.fp16_falcon7b_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_flacon7b_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        quantized_state_dict = quantized_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in quantized_state_dict:
                self.assertTrue(original_params.shape == quantized_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, quantized_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_stablelm_q4_k_m(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.stablelm_model_id,
            gguf_file=self.q4_k_m_stablelm_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.stablelm_model_id, gguf_file=self.q4_k_m_stablelm_model_id)
        text = tokenizer(self.example_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello-\nI am trying to create a new user"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_stablelm_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_stablelm2_model_id,
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.stablelm2_model_id,
            gguf_file=self.fp16_stablelm2_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.stablelm2_model_id, gguf_file=self.fp16_stablelm2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")
        original_out = original_model.generate(**text, max_new_tokens=10)
        converted_out = converted_model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello, I am a 20 year old male"
        self.assertEqual(tokenizer.decode(converted_out[0], skip_special_tokens=True), EXPECTED_TEXT)
        self.assertEqual(
            tokenizer.decode(converted_out[0], skip_special_tokens=True),
            tokenizer.decode(original_out[0], skip_special_tokens=True),
        )

    def test_stablelm_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_stablelm2_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.stablelm2_model_id,
            gguf_file=self.fp16_stablelm2_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_starcoder2_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_original_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_fp16_model_id,
            gguf_file=self.fp16_starcoder2_gguf_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_starcoder2_q6_k(self):
        example_function_text = "def print_hello_world():"
        model = AutoModelForCausalLM.from_pretrained(
            self.starcoder2_model_id,
            gguf_file=self.q6_k_starcoder2_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.starcoder2_model_id, gguf_file=self.q6_k_starcoder2_model_id)
        text = tokenizer(example_function_text, return_tensors="pt").to(torch_device)
        out = model.generate(**text, max_new_tokens=10)

        EXPECTED_TEXT = 'def print_hello_world():\n    print("Hello World")\n\ndef print'
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_mamba_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.mamba_original_model_id,
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.mamba_model_id,
            gguf_file=self.fp16_mamba_model_id,
            torch_dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                if "mixer.A_log" in layer_name:
                    # we should increase tolerance after exponential reversing
                    # and performing np.log(-weights) operation as numbers are slightly different
                    torch.testing.assert_close(original_params, converted_state_dict[layer_name], atol=1e-3, rtol=1e-3)
                else:
                    torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_mamba_q6_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.mamba_model_id,
            gguf_file=self.q6_k_mamba_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.mamba_model_id, gguf_file=self.q6_k_mamba_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello,I answerthe question.\n\nA"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_nemotron_weights_conversion_fp16(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_original_model_id,
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_model_id,
            gguf_file=self.fp16_nemotron_model_id,
            torch_dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_nemotron_q6_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.nemotron_model_id,
            gguf_file=self.q6_k_nemotron_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.nemotron_model_id, gguf_file=self.q6_k_nemotron_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "'Hello. hotmail.com.'"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_q3_k(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.q3_k_gemma2_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.q3_k_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm trying to create a"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_q8_0(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.q8_0_gemma2_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.q8_0_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm a large language model"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    def test_gemma2_fp32(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.fp32_gemma2_model_id,
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.gemma2_model_id, gguf_file=self.fp32_gemma2_model_id)
        text = tokenizer(self.example_text, return_tensors="pt")["input_ids"]
        out = model.generate(text, max_new_tokens=10)

        EXPECTED_TEXT = "Hello! 👋\n\nI'm a large language model"
        self.assertEqual(tokenizer.decode(out[0], skip_special_tokens=True), EXPECTED_TEXT)

    @require_read_token
    def test_gemma2_weights_conversion_fp32(self):
        original_model = AutoModelForCausalLM.from_pretrained(
            self.original_gemma2_model_id,
            torch_dtype=torch.float16,
        )

        converted_model = AutoModelForCausalLM.from_pretrained(
            self.gemma2_model_id,
            gguf_file=self.fp32_gemma2_model_id,
            torch_dtype=torch.float16,
        )

        converted_state_dict = converted_model.state_dict()
        original_state_dict = original_model.state_dict()

        for layer_name, original_params in original_state_dict.items():
            if layer_name in converted_state_dict:
                self.assertTrue(original_params.shape == converted_state_dict[layer_name].shape)
                torch.testing.assert_close(original_params, converted_state_dict[layer_name])
            else:
                raise ValueError(f"Layer {layer_name} is not presented in GGUF model")

    def test_tokenization_xnli(self):
        import tqdm
        from datasets import load_dataset

        gguf_tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        original_tokenizer = AutoTokenizer.from_pretrained(self.original_model_id)

        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = gguf_tokenizer.encode(string)
            encoded2 = original_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("facebook/xnli", "all_languages")

        for i, item in enumerate(tqdm.tqdm(dataset["train"].select(range(100)))):
            for string in item["premise"].values():
                encoded1 = gguf_tokenizer.encode(string)
                encoded2 = original_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2)

                decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)

        # With special tokens
        gguf_tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.q8_0_gguf_model_id)
        original_tokenizer = AutoTokenizer.from_pretrained(self.original_model_id)

        gguf_tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken("<token>", rstrip=False, lstrip=False)]}
        )
        original_tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken("<token>", rstrip=False, lstrip=False)]}
        )

        text = "Hello <token>. <token> Hello"

        encoded1 = gguf_tokenizer.encode(text)
        encoded2 = original_tokenizer.encode(text)

        self.assertEqual(encoded1, encoded2)

        decoded1 = gguf_tokenizer.decode(encoded1, skip_special_tokens=True)
        decoded2 = original_tokenizer.decode(encoded2, skip_special_tokens=True)

        self.assertEqual(decoded1, decoded2)
