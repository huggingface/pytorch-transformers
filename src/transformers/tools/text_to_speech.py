import torch
from datasets import load_dataset

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from .base import PipelineTool


class TextToSpeechTool(PipelineTool):
    pre_processor_class = SpeechT5Processor
    model_class = SpeechT5ForTextToSpeech
    post_processor_class = SpeechT5HifiGan

    description = """
    Text to Speech tool, used to read text out loud. It takes text as input, and outputs a voice signal.
    """

    def encode(self, text, speaker_embeddings=None):
        inputs = self.pre_processor(text=text, return_tensors="pt")

        if speaker_embeddings is None:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    def forward(self, inputs):
        return self.model.generate_speech(**inputs)
