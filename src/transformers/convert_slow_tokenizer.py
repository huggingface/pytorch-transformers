# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
Utilities to convert slow tokenizers in their fast tokenizers counterparts.

All the conversions are grouped here to gather SentencePiece dependencies outside of the fast tokenizers files and
allow to make our dependency on SentencePiece optional.
"""

import warnings
from typing import Dict, List, Tuple

from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece

from .utils import is_protobuf_available, is_sentencepiece_available, logging, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR


logger = logging.get_logger(__name__)


def import_protobuf(error_message=""):
    if is_sentencepiece_available():
        from sentencepiece import sentencepiece_model_pb2

        return sentencepiece_model_pb2
    if is_protobuf_available():
        import google.protobuf

        if version.parse(google.protobuf.__version__) < version.parse("4.0.0"):
            from transformers.utils import sentencepiece_model_pb2
        else:
            from transformers.utils import sentencepiece_model_pb2_new as sentencepiece_model_pb2
        return sentencepiece_model_pb2
    else:
        raise ImportError(PROTOBUF_IMPORT_ERROR.format(error_message))


def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


def generate_merges(vocab, vocab_scores):
    reverse = vocab_scores is not None
    vocab_scores = dict(vocab_scores) if reverse else vocab

    merges = []
    for merge, piece_score in vocab_scores.items():
        local = []
        for index in range(1, len(merge)):
            piece_l, piece_r = merge[:index], merge[index:]
            if piece_l in vocab and piece_r in vocab:
                local.append((piece_l, piece_r, piece_score))
        local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)

    merges = sorted(merges, key=lambda val: (val[2], len(val[0]), len(val[1])), reverse=reverse)
    merges = [(val[0], val[1]) for val in merges]
    return merges


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        requires_backends(self, "sentencepiece")
        from sentencepiece import SentencePieceProcessor

        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        merges = generate_merges(vocab, vocab_scores)

        return vocab, merges


class GemmaSentencePieceExtractor(SentencePieceExtractor):
    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # there is a missing token in the vocab. We have to do this to support merges
        # "<0x09>" is the bytefallback for `\t`
        vocab["\t"] = vocab.get("<0x09>")

        merges = generate_merges(vocab, vocab_scores)
        return vocab, merges


def check_number_comma(piece: str) -> bool:
    return len(piece) < 2 or piece[-1] != "," or not piece[-2].isdigit()


class Converter:
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer

    def converted(self) -> Tokenizer:
        raise NotImplementedError()


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class SplinterConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        question = str(self.original_tokenizer.question_token)
        dot = "."
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id
        question_token_id = self.original_tokenizer.question_token_id
        dot_token_id = self.original_tokenizer.convert_tokens_to_ids(".")

        if self.original_tokenizer.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class FunnelConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:2 $A:0 {sep}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls}:2 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class MPNetConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1",  # MPNet uses two [SEP] tokens
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class OpenAIGPTConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
                fuse_unk=False,
            )
        )

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        return tokenizer


class GPT2Converter(Converter):
    def converted(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None) -> Tokenizer:
        if not vocab:
            vocab = self.original_tokenizer.encoder
        if not merges:
            merges = list(self.original_tokenizer.bpe_ranks)

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        add_prefix_space = getattr(self.original_tokenizer, "add_prefix_space", False)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if getattr(self.original_tokenizer, "add_bos_token", False):
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # XXX trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


class HerbertConverter(Converter):
    def converted(self) -> Tokenizer:
        tokenizer_info_str = "#version:"
        token_suffix = "</w>"

        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        if tokenizer_info_str in merges[0][0]:
            merges = merges[1:]

        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                dropout=None,
                unk_token=self.original_tokenizer.unk_token,
                end_of_word_suffix=token_suffix,
            )
        )

        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix=token_suffix)
        tokenizer.post_processor = processors.BertProcessing(
            sep=(self.original_tokenizer.sep_token, self.original_tokenizer.sep_token_id),
            cls=(self.original_tokenizer.cls_token, self.original_tokenizer.cls_token_id),
        )

        return tokenizer


class Qwen2Converter(Converter):
    def converted(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None) -> Tokenizer:
        if not vocab:
            vocab = self.original_tokenizer.encoder
        if not merges:
            merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                byte_fallback=False,
            )
        )

        tokenizer.normalizer = normalizers.NFC()

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(
                        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                    ),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=getattr(self.original_tokenizer, "add_prefix_space", False),
                    use_regex=False,
                ),
            ]
        )

        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


class RobertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(ot.sep_token, ot.sep_token_id),
            cls=(ot.cls_token, ot.cls_token_id),
            add_prefix_space=ot.add_prefix_space,
            trim_offsets=True,  # True by default on Roberta (historical)
        )

        return tokenizer


class RoFormerConverter(Converter):
    def converted(self) -> Tokenizer:
        from .models.roformer.tokenization_utils import JiebaPreTokenizer

        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        strip_accents = False
        do_lower_case = False
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(JiebaPreTokenizer(vocab))

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class DebertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )

        return tokenizer


class SpmConverter(Converter):
    handle_byte_fallback = False
    SpmExtractor = SentencePieceExtractor
    special_tokens = {}

    def __init__(self, *args):
        requires_backends(self, "protobuf")

        super().__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        if self.proto.trainer_spec.byte_fallback and not self.handle_byte_fallback:
            warnings.warn(
                "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                "unknown tokens into a sequence of byte tokens matching the original piece of text."
            )

    def vocab(self, proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)

        if model_type == 1:
            tokenizer = Tokenizer(
                Unigram(
                    vocab_scores,
                    unk_id=self.unk_id(proto),
                    byte_fallback=self.handle_byte_fallback,
                )
            )

        elif model_type == 2:
            _, merges = self.SpmExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                    byte_fallback=self.handle_byte_fallback,
                    dropout=None,
                )
            )

        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [
            (id, p.piece, p.type == 3 or p.piece in self.special_tokens)
            for id, p in enumerate(proto.pieces)
            if p.type in [3, 4]
        ]
        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=False, special=special)
                for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
            ]
        )

        return tokenizer

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def post_processor(self):
        return None

    def decoder(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


class AlbertConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BarthezConverter(SpmConverter):
    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class CamembertConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>NOTUSED", 0.0),
            ("<pad>", 0.0),
            ("</s>NOTUSED", 0.0),
            ("<unk>", 0.0),
            ("<unk>NOTUSED", -100),
        ]
        # We down-grade the original SentencePiece by -100 to avoid using it and use our added token instead
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[1:]]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        # See vocab unk position
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class DebertaV2Converter(SpmConverter):
    def pre_tokenizer(self, replacement, add_prefix_space):
        list_pretokenizers = []
        if self.original_tokenizer.split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme))
        return pre_tokenizers.Sequence(list_pretokenizers)

    def normalizer(self, proto):
        list_normalizers = []
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        list_normalizers.append(normalizers.Strip())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class MBartConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [
            ("ar_AR", 0.0),
            ("cs_CZ", 0.0),
            ("de_DE", 0.0),
            ("en_XX", 0.0),
            ("es_XX", 0.0),
            ("et_EE", 0.0),
            ("fi_FI", 0.0),
            ("fr_XX", 0.0),
            ("gu_IN", 0.0),
            ("hi_IN", 0.0),
            ("it_IT", 0.0),
            ("ja_XX", 0.0),
            ("kk_KZ", 0.0),
            ("ko_KR", 0.0),
            ("lt_LT", 0.0),
            ("lv_LV", 0.0),
            ("my_MM", 0.0),
            ("ne_NP", 0.0),
            ("nl_XX", 0.0),
            ("ro_RO", 0.0),
            ("ru_RU", 0.0),
            ("si_LK", 0.0),
            ("tr_TR", 0.0),
            ("vi_VN", 0.0),
            ("zh_CN", 0.0),
        ]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="$A </s> en_XX",
            pair="$A $B </s> en_XX",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class MBart50Converter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [("ar_AR", 0.0), ("cs_CZ", 0.0), ("de_DE", 0.0), ("en_XX", 0.0), ("es_XX", 0.0), ("et_EE", 0.0), ("fi_FI", 0.0), ("fr_XX", 0.0), ("gu_IN", 0.0), ("hi_IN", 0.0), ("it_IT", 0.0), ("ja_XX", 0.0), ("kk_KZ", 0.0), ("ko_KR", 0.0), ("lt_LT", 0.0), ("lv_LV", 0.0), ("my_MM", 0.0), ("ne_NP", 0.0), ("nl_XX", 0.0), ("ro_RO", 0.0), ("ru_RU", 0.0), ("si_LK", 0.0), ("tr_TR", 0.0), ("vi_VN", 0.0), ("zh_CN", 0.0), ("af_ZA", 0.0), ("az_AZ", 0.0), ("bn_IN", 0.0), ("fa_IR", 0.0), ("he_IL", 0.0), ("hr_HR", 0.0), ("id_ID", 0.0), ("ka_GE", 0.0), ("km_KH", 0.0), ("mk_MK", 0.0), ("ml_IN", 0.0), ("mn_MN", 0.0), ("mr_IN", 0.0), ("pl_PL", 0.0), ("ps_AF", 0.0), ("pt_XX", 0.0), ("sv_SE", 0.0), ("sw_KE", 0.0), ("ta_IN", 0.0), ("te_IN", 0.0), ("th_TH", 0.0), ("tl_XX", 0.0), ("uk_UA", 0.0), ("ur_PK", 0.0), ("xh_ZA", 0.0), ("gl_ES", 0.0), ("sl_SI", 0.0)]  # fmt: skip
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="en_XX $A </s>",
            pair="en_XX $A $B </s>",
            special_tokens=[
                ("en_XX", self.original_tokenizer.convert_tokens_to_ids("en_XX")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class NllbConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        return 3

    def post_processor(self):
        return processors.TemplateProcessing(
            single="eng_Latn $A </s>",
            pair="eng_Latn $A $B </s>",
            special_tokens=[
                ("eng_Latn", self.original_tokenizer.convert_tokens_to_ids("eng_Latn")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class SeamlessM4TConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<pad>", 0.0),
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        return self.original_tokenizer.unk_token_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="__eng__ $A </s>",
            pair="__eng__ $A $B </s>",
            special_tokens=[
                ("__eng__", self.original_tokenizer.convert_tokens_to_ids("__eng__")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class XLMRobertaConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [("<mask>", 0.0)]
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        return [
            (piece.piece, piece.score) if check_number_comma(piece.piece) else (piece.piece, piece.score - 100)
            for piece in proto.pieces
        ]

    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        list_normalizers.append(normalizers.Replace(Regex(" {2,}"), " "))
        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="$A:0 <sep>:0 <cls>:2",
            pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
            special_tokens=[
                ("<sep>", self.original_tokenizer.convert_tokens_to_ids("<sep>")),
                ("<cls>", self.original_tokenizer.convert_tokens_to_ids("<cls>")),
            ],
        )


class ReformerConverter(SpmConverter):
    pass


class RemBertConverter(SpmConverter):
    # Inspired from AlbertConverter
    def normalizer(self, proto):
        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.original_tokenizer.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.original_tokenizer.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap

        if precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))

        return normalizers.Sequence(list_normalizers)

    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class BertGenerationConverter(SpmConverter):
    pass


class PegasusConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
        ]

        if self.original_tokenizer.mask_token_sent is not None:
            vocab += [(self.original_tokenizer.mask_token_sent, 0.0)]

        if (
            self.original_tokenizer.mask_token is not None
            and self.original_tokenizer.mask_token_id < self.original_tokenizer.offset
        ):
            vocab += [(self.original_tokenizer.mask_token, 0.0)]

        vocab += [(f"<unk_{i}>", -100.0) for i in range(2, self.original_tokenizer.offset)]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[2:]]
        return vocab

    def unk_id(self, proto):
        return proto.trainer_spec.unk_id + self.original_tokenizer.offset

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme),
            ]
        )

    def post_processor(self):
        eos = self.original_tokenizer.eos_token
        special_tokens = [
            (eos, self.original_tokenizer.eos_token_id),
        ]
        return processors.TemplateProcessing(single=["$A", eos], pair=["$A", "$B", eos], special_tokens=special_tokens)


class T5Converter(SpmConverter):
    def vocab(self, proto):
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        vocab += [(f"<extra_id_{i}>", 0.0) for i in range(num_extra_ids - 1, -1, -1)]
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class UdopConverter(SpmConverter):
    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class WhisperConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=self.original_tokenizer.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        prefix_token_ids = self.original_tokenizer.prefix_tokens
        prefixes = self.original_tokenizer.convert_ids_to_tokens(prefix_token_ids)
        eos = self.original_tokenizer.eos_token
        eos_token_id = self.original_tokenizer.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

        return tokenizer


class BigBirdConverter(SpmConverter):
    def post_processor(self):
        return processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.original_tokenizer.convert_tokens_to_ids("[CLS]")),
                ("[SEP]", self.original_tokenizer.convert_tokens_to_ids("[SEP]")),
            ],
        )


class CLIPConverter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.encoder
        merges = list(self.original_tokenizer.bpe_ranks.keys())
        unk_token = self.original_tokenizer.unk_token

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""),
                    behavior="removed",
                    invert=True,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()

        # Hack to have a ByteLevel and TemplaceProcessor
        tokenizer.post_processor = processors.RobertaProcessing(
            sep=(self.original_tokenizer.eos_token, self.original_tokenizer.eos_token_id),
            cls=(self.original_tokenizer.bos_token, self.original_tokenizer.bos_token_id),
            add_prefix_space=False,
            trim_offsets=False,
        )
        return tokenizer


class LayoutLMv2Converter(Converter):
    def converted(self) -> Tokenizer:
        vocab = self.original_tokenizer.vocab
        tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(self.original_tokenizer.unk_token)))

        tokenize_chinese_chars = False
        strip_accents = False
        do_lower_case = True
        if hasattr(self.original_tokenizer, "basic_tokenizer"):
            tokenize_chinese_chars = self.original_tokenizer.basic_tokenizer.tokenize_chinese_chars
            strip_accents = self.original_tokenizer.basic_tokenizer.strip_accents
            do_lower_case = self.original_tokenizer.basic_tokenizer.do_lower_case

        tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=f"{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")

        return tokenizer


class BlenderbotConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {ot.eos_token}:0",
            special_tokens=[
                (ot.eos_token, ot.eos_token_id),
            ],
        )

        return tokenizer


class XGLMConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [
            ("<s>", 0.0),
            ("<pad>", 0.0),
            ("</s>", 0.0),
            ("<unk>", 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        vocab += [("<madeupword0>", 0.0), ("<madeupword1>", 0.0), ("<madeupword2>", 0.0), ("<madeupword3>", 0.0), ("<madeupword4>", 0.0), ("<madeupword5>", 0.0), ("<madeupword6>", 0.0)]  # fmt: skip
        return vocab

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def post_processor(self):
        return processors.TemplateProcessing(
            single="</s> $A",
            pair="</s> $A </s> </s> $B",
            special_tokens=[
                ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


class GemmaConverter(SpmConverter):
    handle_byte_fallback = True
    SpmExtractor = GemmaSentencePieceExtractor
    # start and end of turn tokens must be marked as special
    special_tokens = {"<start_of_turn>", "<end_of_turn>"}

    """"
    split_by_unicode_script: true
    split_by_number: true
    split_by_whitespace: true
    treat_whitespace_as_suffix: false
    allow_whitespace_only_pieces: true
    split_digits: true
    byte_fallback: true
    """

    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
            (self.original_tokenizer.bos_token, 0.0),
        ]
        for piece in proto.pieces[3:]:
            if piece.piece == "<0x09>":
                vocab += [("\t", piece.score)]
            else:
                vocab += [(piece.piece, piece.score)]
        # vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Split(" ", "merged_with_previous")

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )


class LlamaConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.convert_ids_to_tokens(0), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(1), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(2), 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def normalizer(self, proto):
        if getattr(self.original_tokenizer, "legacy", True):
            sequence = []
            if getattr(self.original_tokenizer, "add_prefix_space", True):
                sequence += [normalizers.Prepend(prepend="▁")]
            sequence += [normalizers.Replace(pattern=" ", content="▁")]
            return normalizers.Sequence(sequence)
        return None  # non-legacy, no normalizer

    def pre_tokenizer(self, replacement, add_prefix_space):
        if not getattr(self.original_tokenizer, "legacy", True):  # non-legacy, we need a replace
            prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
            return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)
        return None

    def post_processor(self):
        # the processor is defined in the LlamaTokenizerFast class.
        return None


class MarkupLMConverter(Converter):
    def converted(self) -> Tokenizer:
        ot = self.original_tokenizer
        vocab = ot.encoder
        merges = list(ot.bpe_ranks.keys())

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
                unk_token=self.original_tokenizer.unk_token,
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ot.add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()

        cls = str(self.original_tokenizer.cls_token)
        sep = str(self.original_tokenizer.sep_token)
        cls_token_id = self.original_tokenizer.cls_token_id
        sep_token_id = self.original_tokenizer.sep_token_id

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
            ],
        )

        return tokenizer


class MoshiConverter(SpmConverter):
    handle_byte_fallback = True

    def __init__(self, vocab_file, model_max_length=None, **kwargs):
        requires_backends(self, "protobuf")

        Converter.__init__(self, vocab_file)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        with open(vocab_file, "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Replace(" ", "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = "first"
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class TikTokenConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab_file=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab_file = vocab_file
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, tiktoken_url: str):
        try:
            from tiktoken.load import load_tiktoken_bpe
        except Exception:
            raise ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with " "`pip install tiktoken`."
            )

        bpe_ranks = load_tiktoken_bpe(tiktoken_url)
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab_file)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


SLOW_TO_FAST_CONVERTERS = {
    "AlbertTokenizer": AlbertConverter,
    "BartTokenizer": RobertaConverter,
    "BarthezTokenizer": BarthezConverter,
    "BertTokenizer": BertConverter,
    "BigBirdTokenizer": BigBirdConverter,
    "BlenderbotTokenizer": BlenderbotConverter,
    "CamembertTokenizer": CamembertConverter,
    "CLIPTokenizer": CLIPConverter,
    "CodeGenTokenizer": GPT2Converter,
    "ConvBertTokenizer": BertConverter,
    "DebertaTokenizer": DebertaConverter,
    "DebertaV2Tokenizer": DebertaV2Converter,
    "DistilBertTokenizer": BertConverter,
    "DPRReaderTokenizer": BertConverter,
    "DPRQuestionEncoderTokenizer": BertConverter,
    "DPRContextEncoderTokenizer": BertConverter,
    "ElectraTokenizer": BertConverter,
    "FNetTokenizer": AlbertConverter,
    "FunnelTokenizer": FunnelConverter,
    "GPT2Tokenizer": GPT2Converter,
    "HerbertTokenizer": HerbertConverter,
    "LayoutLMTokenizer": BertConverter,
    "LayoutLMv2Tokenizer": BertConverter,
    "LayoutLMv3Tokenizer": RobertaConverter,
    "LayoutXLMTokenizer": XLMRobertaConverter,
    "LongformerTokenizer": RobertaConverter,
    "LEDTokenizer": RobertaConverter,
    "LxmertTokenizer": BertConverter,
    "MarkupLMTokenizer": MarkupLMConverter,
    "MBartTokenizer": MBartConverter,
    "MBart50Tokenizer": MBart50Converter,
    "MPNetTokenizer": MPNetConverter,
    "MobileBertTokenizer": BertConverter,
    "MvpTokenizer": RobertaConverter,
    "NllbTokenizer": NllbConverter,
    "OpenAIGPTTokenizer": OpenAIGPTConverter,
    "PegasusTokenizer": PegasusConverter,
    "Qwen2Tokenizer": Qwen2Converter,
    "RealmTokenizer": BertConverter,
    "ReformerTokenizer": ReformerConverter,
    "RemBertTokenizer": RemBertConverter,
    "RetriBertTokenizer": BertConverter,
    "RobertaTokenizer": RobertaConverter,
    "RoFormerTokenizer": RoFormerConverter,
    "SeamlessM4TTokenizer": SeamlessM4TConverter,
    "SqueezeBertTokenizer": BertConverter,
    "T5Tokenizer": T5Converter,
    "UdopTokenizer": UdopConverter,
    "WhisperTokenizer": WhisperConverter,
    "XLMRobertaTokenizer": XLMRobertaConverter,
    "XLNetTokenizer": XLNetConverter,
    "SplinterTokenizer": SplinterConverter,
    "XGLMTokenizer": XGLMConverter,
    "LlamaTokenizer": LlamaConverter,
    "CodeLlamaTokenizer": LlamaConverter,
    "GemmaTokenizer": GemmaConverter,
    "Phi3Tokenizer": LlamaConverter,
}


def convert_slow_tokenizer(transformer_tokenizer, from_tiktoken=False) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].
       from_tiktoken (bool, optional): Whether to use the `tiktoken` library to convert the tokenizer instead of sentencepiece.
            Defaults to False.

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    if tokenizer_class_name in SLOW_TO_FAST_CONVERTERS and not from_tiktoken:
        converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]
        return converter_class(transformer_tokenizer).converted()

    else:
        try:
            logger.info("Converting from Tiktoken")
            return TikTokenConverter(
                vocab_file=transformer_tokenizer.vocab_file,
                additional_special_tokens=transformer_tokenizer.additional_special_tokens,
            ).converted()
        except Exception:
            raise ValueError(
                f"Converting from Tiktoken failed, if a converter for SentencePiece is available, provide a model path "
                f"with a SentencePiece tokenizer.model file."
                f"Currently available slow->fast convertors: {list(SLOW_TO_FAST_CONVERTERS.keys())}"
            )
