# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import copy
import tempfile
import unittest

import numpy as np

from transformers import BartConfig, BartTokenizer, is_tf_available
from transformers.testing_utils import require_tf, slow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...utils.test_modeling_tf_core import TFCoreModelTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFBartForConditionalGeneration, TFBartForSequenceClassification, TFBartModel


@require_tf
class TFBartModelTester:
    config_cls = BartConfig
    config_updates = {}
    hidden_act = "gelu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        # Ids are clipped to avoid "beginng of sequence", "end of sequence", and "pad" tokens
        input_ids = tf.clip_by_value(
            ids_tensor([self.batch_size, self.seq_length - 1], self.vocab_size),
            clip_value_min=self.eos_token_id + 1,
            clip_value_max=self.vocab_size + 1,
        )
        # Explicity add "end of sequence" to the inputs
        eos_tensor = tf.expand_dims(tf.constant([self.eos_token_id] * self.batch_size), 1)
        input_ids = tf.concat([input_ids, eos_tensor], axis=1)

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_ids=[2],
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.pad_token_id,
            **self.config_updates,
        )
        inputs_dict = prepare_bart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = TFBartModel(config=config).get_decoder()
        input_ids = inputs_dict["input_ids"]

        input_ids = input_ids[:1, :]
        attention_mask = inputs_dict["attention_mask"][:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = tf.cast(ids_tensor((self.batch_size, 3), 2), tf.int8)

        # append to next input_ids and
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([attention_mask, next_attn_mask], axis=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)
        output_from_no_past = output_from_no_past[0]

        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)
        output_from_past = output_from_past[0]

        self.parent.assertEqual(next_tokens.shape[1], output_from_past.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), output_from_past.shape[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)


def prepare_bart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_ids, config.pad_token_id), tf.int8)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.concat(
            [
                tf.ones(decoder_input_ids[:, :1].shape, dtype=tf.int8),
                tf.cast(tf.math.not_equal(decoder_input_ids[:, 1:], config.pad_token_id), tf.int8),
            ],
            axis=-1,
        )
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }


@require_tf
class TFBartModelTest(TFModelTesterMixin, TFCoreModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFBartForConditionalGeneration, TFBartForSequenceClassification, TFBartModel) if is_tf_available() else ()
    )
    all_generative_model_classes = (TFBartForConditionalGeneration,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": TFBartModel,
            "summarization": TFBartForConditionalGeneration,
            "text-classification": TFBartForSequenceClassification,
            "text2text-generation": TFBartForConditionalGeneration,
            "translation": TFBartForConditionalGeneration,
            "zero-shot": TFBartForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    is_encoder_decoder = True
    test_pruning = False
    test_onnx = True
    onnx_min_opset = 10

    def setUp(self):
        self.model_tester = TFBartModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_decoder_model_past_large_inputs(*config_and_inputs)

    # TODO (Joao): fix me
    @unittest.skip("Onnx compliancy broke with TF 2.10")
    def test_onnx_compliancy(self):
        pass

    # TFBartForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (TFBartForConditionalGeneration, TFBartModel):
            model = model_class(config)

            inputs = copy.deepcopy(inputs_dict)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
            else:
                inputs["inputs_embeds"] = model.get_input_embeddings()(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = model.get_input_embeddings()(decoder_input_ids)

            inputs = self._prepare_for_class(inputs, model_class)

            model(inputs)

    # TFBartForSequenceClassification does not support inputs_embeds
    @slow
    def test_graph_mode_with_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (TFBartForConditionalGeneration, TFBartModel):
            model = model_class(config)

            inputs = copy.deepcopy(inputs_dict)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
            else:
                inputs["inputs_embeds"] = model.get_input_embeddings()(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = model.get_input_embeddings()(decoder_input_ids)

            inputs = self._prepare_for_class(inputs, model_class)

            @tf.function
            def run_in_graph_mode():
                return model(inputs)

            outputs = run_in_graph_mode()
            self.assertIsNotNone(outputs)

    @slow
    def test_save_load_after_resize_token_embeddings(self):
        # Custom version of this test to ensure "end of sequence" tokens are present throughout
        if not self.test_resize_embeddings:
            return
        config, original_inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # create a model with resized (expended) embeddings
            new_tokens_size = 10
            old_total_size = config.vocab_size
            new_total_size = old_total_size + new_tokens_size
            model = model_class(config=copy.deepcopy(config))  # `resize_token_embeddings` mutates `config`
            model.build_in_name_scope()
            model.resize_token_embeddings(new_total_size)

            # fetch the output for an input exclusively made of new members of the vocabulary
            inputs_dict = copy.deepcopy(original_inputs_dict)
            ids_feat_name = None
            if "input_ids" in inputs_dict:
                ids_feat_name = "input_ids"
            elif "decoder_input_ids" in inputs_dict:
                ids_feat_name = "decoder_input_ids"
            else:
                assert False, "No input ids feature found in the inputs dict"

            new_vocab_input_ids = ids_tensor(inputs_dict[ids_feat_name].shape, new_tokens_size)
            new_vocab_input_ids += old_total_size

            # Replace last id with EOS token
            new_vocab_input_ids = new_vocab_input_ids[:, :-1]
            new_vocab_input_ids = tf.concat(
                [new_vocab_input_ids, tf.ones((tf.shape(new_vocab_input_ids)[0], 1), dtype=tf.int32) * 2], axis=1
            )

            inputs_dict[ids_feat_name] = new_vocab_input_ids
            if "input_ids" in inputs_dict:
                inputs_dict["input_ids"] = new_vocab_input_ids
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"] = new_vocab_input_ids
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)
            outputs = model(**prepared_inputs)

            # save and load the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)
                restored_model_outputs = model(**prepared_inputs)

                # check that the output for the restored model is the same
                self.assert_outputs_same(restored_model_outputs, outputs)


def _long_tensor(tok_lst):
    return tf.constant(tok_lst, dtype=tf.int32)


@require_tf
class TFBartHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        eos_column_vector = tf.ones((4, 1), dtype=tf.int32) * 2
        input_ids = tf.concat([ids_tensor((4, 6), self.vocab_size - 3) + 3, eos_column_vector], axis=1)
        batch_size = input_ids.shape[0]
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            decoder_start_token_id=2,
        )
        return config, input_ids, batch_size

    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        decoder_lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size)
        lm_model = TFBartForConditionalGeneration(config)
        outputs = lm_model(input_ids=input_ids, labels=decoder_lm_labels, decoder_input_ids=input_ids, use_cache=False)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape)

    def test_lm_uneven_forward(self):
        config = BartConfig(
            vocab_size=10,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
        )
        lm_model = TFBartForConditionalGeneration(config)
        context = tf.fill((7, 2), 4)
        summary = tf.fill((7, 7), 6)
        outputs = lm_model(input_ids=context, decoder_input_ids=summary, use_cache=False)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape)


@require_tf
class TFBartForSequenceClassificationTest(unittest.TestCase):
    def test_model_fails_for_uneven_eos_tokens(self):
        config = BartConfig(eos_token_id=2)
        model = TFBartForSequenceClassification(config)
        inputs = {
            "input_ids": tf.constant([[1, 2, 2, 2], [1, 3, 2, 2], [2, 2, 3, 3]]),
            "attention_mask": tf.constant([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        }
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model(inputs)


@slow
@require_tf
class TFBartModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head(self):
        model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large").model

        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        attention_mask = tf.cast(tf.math.not_equal(input_ids, model.config.pad_token_id), tf.int8)
        output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 1024)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = tf.convert_to_tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]],
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=1e-3)

    def test_cnn_summarization_same_as_fairseq_hard(self):
        hf = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tok = self.tok

        FRANCE_ARTICLE = (  # @noqa
            " Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
            " Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane."
            ' Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation."'
            ' He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s'
            " comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
            " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French"
            " Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a"
            " phone at the wreckage site. The two publications described the supposed video, but did not post it on"
            " their websites. The publications said that they watched the video, which was found by a source close to"
            " the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported."
            ' "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the'
            " cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the"
            ' screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt,'
            " editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said"
            " the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman"
            " in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the"
            ' reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said,'
            ' but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be'
            " sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by"
            " specialized technicians working hand-in-hand with investigators. But none of the cell phones found so"
            " far have been sent to the institute, Menichini said. Asked whether staff involved in the search could"
            ' have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin'
            ' Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match'
            ' are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
            ' cell phones from the crash site after Bild and Paris Match published their reports. "That is something'
            " we did not know before. ... Overall we can say many things of the investigation weren't revealed by the"
            ' investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline'
            " Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the"
            " controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the"
            ' French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of'
            ' severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school'
            " discovered in an internal investigation, Lufthansa said, included medical documents he submitted in"
            " connection with resuming his flight training. The announcement indicates that Lufthansa, the parent"
            " company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and"
            " ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100%"
            ' fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was'
            " sharing the information and documents -- including training and medical records -- with public"
            " prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the"
            " past week to recover human remains and plane debris scattered across a steep mountainside. He saw the"
            " crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash"
            " site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late"
            " Tuesday that no visible human remains were left at the site but recovery teams would keep searching."
            " French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all"
            " the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
            " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said."
            " Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew"
            " on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with"
            " the flight school during his training were among several developments as investigators continued to"
            " delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa"
            " spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his"
            ' examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in'
            " Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at"
            " some point before his aviation career and underwent psychotherapy before he got his pilot's license."
            " Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the"
            " crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to"
            " lose his pilot's license, a European government official briefed on the investigation told CNN on"
            ' Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being'
            " considered. Another source, a law enforcement official briefed on the investigation, also told CNN that"
            " authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would"
            " not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had"
            " seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded"
            " he had psychological issues, the European government official said. But no matter what details emerge"
            " about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
            ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact'
            " that maybe they weren't going to keep doing their job and they're upset about that and so they're"
            ' suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to'
            " also take that rage and turn it outward on 149 other people who had nothing to do with the person's"
            ' problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight'
            " 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura"
            " Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine"
            " Amiel and Anna-Maja Rappard contributed to this report."
        )
        EXPECTED_SUMMARY_FRANCE = (
            "French prosecutor says he's not aware of any video footage from on board the plane. German daily Bild"
            " and French Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie"
            ' spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa confirms'
            " co-pilot Andreas Lubitz had battled depression."
        )

        SHORTER_ARTICLE = (
            " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
            " and Faith Karimi contributed to this report."
        )
        EXPECTED_SUMMARY_SHORTER = (
            "The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives"
            " the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States"
            " opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said"
            " it was a move toward greater justice."
        )

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        IRAN_ARTICLE = (
            " (CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
            " in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively"
            " block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger."
            " Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli"
            " Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a"
            " letter to the Iranian leadership warning them away from a deal. The debate that has already begun since"
            " the announcement of the new framework will likely result in more heat than light. It will not be helped"
            " by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: ."
            " The most misleading assertion, despite universal rejection by experts, is that the negotiations'"
            " objective at the outset was the total elimination of any nuclear program in Iran. That is the position"
            " of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it"
            " had been, there would have been no Iranian team at the negotiating table. Rather, the objective has"
            " always been to structure an agreement or series of agreements so that Iran could not covertly develop a"
            " nuclear arsenal before the United States and its allies could respond. The new framework has exceeded"
            " expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by"
            " two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another"
            " dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite"
            " sharp accusations by some in the United States and its allies, Iran denies having such a program, and"
            " U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's"
            " continued cooperation with International Atomic Energy Agency inspections is further evidence on this"
            " point, and we'll know even more about Iran's program in the coming months and years because of the deal."
            " In fact, the inspections provisions that are part of this agreement are designed to protect against any"
            " covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that"
            " the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter"
            " warning that a deal might be killed by Congress or a future president). This of course is not the case."
            " The talks were between Iran and the five permanent members of the U.N. Security Council (United States,"
            " United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has"
            " played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement"
            " reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran"
            " and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement"
            " contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the"
            " case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased"
            " or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes"
            " Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear"
            " sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going"
            " forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such"
            " a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the"
            ' agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not'
            " suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New"
            " START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement"
            " with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement"
            " will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove"
            " most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally"
            " some insist that any agreement must address Iranian missile programs, human rights violations or support"
            " for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are"
            " unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in"
            " the negotiations would be a poison pill. This agreement should be judged on its merits and on how it"
            " affects the security of our negotiating partners and allies, including Israel. Those judgments should be"
            " fact-based, not based on questionable assertions or dubious assumptions."
        )
        EXPECTED_SUMMARY_IRAN = (
            "The U.S. and its negotiating partners reached a very strong framework agreement with Iran. Peter Bergen:"
            " The debate that has already begun will likely result in more heat than light. He says the agreement"
            " limits Iran's nuclear program in such a way as to effectively block it from building a nuclear weapon."
            " Bergen says the most important aim of a nuclear deal is preventing a nuclear Iran."
        )

        ARTICLE_SUBWAY = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )
        EXPECTED_SUMMARY_SUBWAY = (
            "Liana Barrientos has been married 10 times, sometimes within two weeks of each other. Prosecutors say the"
            " marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in"
            " the Bronx. She was arrested and charged with theft of service and criminal trespass for allegedly"
            " sneaking into the subway."
        )

        dct = tok(
            [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY],
            max_length=1024,
            truncation_strategy="only_first",
            padding="longest",
            truncation=True,
            return_tensors="tf",
        )
        self.assertEqual(1024, dct["input_ids"].shape[1])
        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
        )

        assert hypotheses_batch[:, 1].numpy().tolist() == [0, 0, 0, 0]  # test force_bos_token_to_be_generated
        decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        expected_batch = [
            EXPECTED_SUMMARY_FRANCE,
            EXPECTED_SUMMARY_SHORTER,
            EXPECTED_SUMMARY_IRAN,
            EXPECTED_SUMMARY_SUBWAY,
        ]
        assert decoded == expected_batch

    @cached_property
    def tok(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @slow
    def test_contrastive_search_bart(self):
        article = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        input_ids = bart_tokenizer(
            article, add_special_tokens=False, truncation=True, max_length=512, return_tensors="tf"
        ).input_ids

        outputs = bart_model.generate(input_ids, penalty_alpha=0.5, top_k=5, max_length=64)
        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "Liana Barrientos, 39, pleaded not guilty to charges related to false marriage statements. "
                "Prosecutors say she married at least 10 times, sometimes within two weeks of each other. She is "
                "accused of being part of an immigration scam to get permanent residency. If convicted, she faces up "
                "to four years in"
            ],
        )

    @slow
    def test_contrastive_search_bart_xla(self):
        article = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        input_ids = bart_tokenizer(
            article, add_special_tokens=False, truncation=True, max_length=512, return_tensors="tf"
        ).input_ids

        xla_generate = tf.function(bart_model.generate, jit_compile=True)
        # no_repeat_ngram_size set to 0 because it isn't compatible with XLA, but doesn't change the original output
        outputs = xla_generate(input_ids, penalty_alpha=0.5, top_k=5, max_length=64, no_repeat_ngram_size=0)
        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "Liana Barrientos, 39, pleaded not guilty to charges related to false marriage statements. "
                "Prosecutors say she married at least 10 times, sometimes within two weeks of each other. She is "
                "accused of being part of an immigration scam to get permanent residency. If convicted, she faces up "
                "to four years in"
            ],
        )


@slow
@require_tf
class FasterTFBartModelIntegrationTests(unittest.TestCase):
    """These tests are useful for debugging since they operate on a model with 1 encoder layer and 1 decoder layer."""

    @cached_property
    def tok(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @cached_property
    def xsum_1_1_model(self):
        return TFBartForConditionalGeneration.from_pretrained("sshleifer/distilbart-xsum-1-1")

    def test_xsum_1_1_generation(self):
        model = self.xsum_1_1_model
        assert model.model.decoder.embed_tokens == model.model.shared
        ARTICLE = (
            "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes."
        )
        EXPECTED = (
            " The International Criminal Court (ICC) has announced that it has been announced by the International"
            " Criminal court."
        )
        dct = self.tok(ARTICLE, return_tensors="tf")
        generated_ids = model.generate(**dct, num_beams=4)
        result = self.tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert result == EXPECTED

    def test_xsum_1_1_xla_generation(self):
        # same test as above, but with `no_repeat_ngram_size=0` (not compatible with XLA) and XLA comparison enabled
        model = self.xsum_1_1_model
        assert model.model.decoder.embed_tokens == model.model.shared
        ARTICLE = (
            "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes."
        )
        EXPECTED = (
            " The International Criminal Court (ICC) has announced that it is to be investigated by the International"
            " Criminal Court (ICC) over allegations of war crimes."
        )

        dct = self.tok(ARTICLE, return_tensors="tf")
        generated_ids = model.generate(**dct, num_beams=4, no_repeat_ngram_size=0)
        result = self.tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert result == EXPECTED

        xla_generate = tf.function(model.generate, jit_compile=True)
        generated_ids = xla_generate(**dct, num_beams=4, no_repeat_ngram_size=0)
        result = self.tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert result == EXPECTED

    def test_xsum_1_1_batch_generation(self):
        batch = self.tok(
            [
                "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
                " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
                " The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is"
                " based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted"
                ' its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including'
                ' East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination'
                " into the situation in Palestinian territories, paving the way for possible war crimes investigations"
                " against Israelis. As members of the court, Palestinians may be subject to counter-charges as well."
                " Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts"
                " to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony,"
                ' said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome'
                ' Statute today, the world is also a step closer to ending a long era of impunity and injustice," he'
                ' said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of'
                ' justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was'
                ' just the first step for the Palestinians. "As the Rome Statute today enters into force for the State'
                " of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a"
                ' State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she'
                ' said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize'
                " Palestine for joining the ICC should immediately end their pressure, and countries that support"
                " universal acceptance of the court's treaty should speak out to welcome its membership,\" said"
                " Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts"
                " to undermine international justice, not Palestine's decision to join a treaty to which over 100"
                ' countries around the world are members." In January, when the preliminary ICC examination was'
                " opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was"
                ' overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s'
                ' decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we'
                ' do not believe that it is eligible to join the ICC," the State Department said in a statement. It'
                ' urged the warring sides to resolve their differences through direct negotiations. "We will continue'
                ' to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said.'
                " But the ICC begs to differ with the definition of a state for its purposes and refers to the"
                ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows'
                " the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor"
                ' Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality."'
                " The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The"
                " inquiry will include alleged war crimes committed since June. The International Criminal Court was"
                " set up in 2002 to prosecute genocide, crimes against humanity and war crimes.",
                "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted"
                " Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor"
                ' Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A'
                " person who has such a video needs to immediately give it to the investigators.\" Robin's comments"
                " follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
                " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the"
                " French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was"
                " recovered from a phone at the wreckage site. The two publications described the supposed video, but"
                " did not post it on their websites. The publications said that they watched the video, which was"
                " found by a source close to the investigation. \"One can hear cries of 'My God' in several"
                ' languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps'
                " of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy"
                ' shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing'
                " scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident"
                " investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc"
                " Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the"
                ' Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell'
                ' phones have been collected at the site, he said, but that they "hadn\'t been exploited yet."'
                " Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute"
                " in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working"
                " hand-in-hand with investigators. But none of the cell phones found so far have been sent to the"
                " institute, Menichini said. Asked whether staff involved in the search could have leaked a memory"
                ' card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett:'
                ' Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are'
                ' "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
                ' cell phones from the crash site after Bild and Paris Match published their reports. "That is'
                " something we did not know before. ... Overall we can say many things of the investigation weren't"
                ' revealed by the investigation at the beginning," he said. What was mental state of Germanwings'
                " co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled"
                " depression years before he took the controls of Germanwings Flight 9525, which he's accused of"
                " deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school"
                ' in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email'
                " correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa"
                " said, included medical documents he submitted in connection with resuming his flight training. The"
                " announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle"
                " with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa,"
                " whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday"
                ' as a "swift and seamless clarification" and said it was sharing the information and documents --'
                " including training and medical records -- with public prosecutors. Spohr traveled to the crash site"
                " Wednesday, where recovery teams have been working for the past week to recover human remains and"
                " plane debris scattered across a steep mountainside. He saw the crisis center set up in"
                " Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving"
                " families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no"
                " visible human remains were left at the site but recovery teams would keep searching. French"
                " President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the"
                " victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
                " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini"
                " said. Among those personal belongings could be more cell phones belonging to the 144 passengers and"
                " six crew on board. Check out the latest from our correspondents . The details about Lubitz's"
                " correspondence with the flight school during his training were among several developments as"
                " investigators continued to delve into what caused the crash and Lubitz's possible motive for"
                " downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical"
                ' certificate, had passed all his examinations and "held all the licenses required." Earlier, a'
                " spokesman for the prosecutor's office in Dusseldorf, Christoph Kumpa, said medical records reveal"
                " Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent"
                " psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting"
                " Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether"
                " Lubitz feared his medical condition would cause him to lose his pilot's license, a European"
                ' government official briefed on the investigation told CNN on Tuesday. While flying was "a big part'
                " of his life,\" the source said, it's only one theory being considered. Another source, a law"
                " enforcement official briefed on the investigation, also told CNN that authorities believe the"
                " primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly"
                " because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor"
                " and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had"
                " psychological issues, the European government official said. But no matter what details emerge about"
                " his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
                ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the'
                " fact that maybe they weren't going to keep doing their job and they're upset about that and so"
                ' they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels'
                " entitled to also take that rage and turn it outward on 149 other people who had nothing to do with"
                " the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of"
                " Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from"
                " Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff,"
                " Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.",
            ],
            return_tensors="tf",
            padding="longest",
            truncation=True,
        )
        generated_ids = self.xsum_1_1_model.generate(**batch, num_beams=4)
        result = self.tok.batch_decode(generated_ids, skip_special_tokens=True)
        assert (
            result[0]
            == " The International Criminal Court (ICC) has announced that it has been announced by the International"
            " Criminal court."
        )
        assert (
            result[1]
            == " An investigation into the crash that killed at least 10 people in the French capital has been"
            " released by the French police investigating the crash."
        )

    def test_encoder_equiv(self):
        batch = self.tok(
            [
                "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
                " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
                " The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is"
                " based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted"
                ' its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including'
                ' East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination'
                " into the situation in Palestinian territories, paving the way for possible war crimes investigations"
                " against Israelis. As members of the court, Palestinians may be subject to counter-charges as well."
                " Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts"
                " to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony,"
                ' said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome'
                ' Statute today, the world is also a step closer to ending a long era of impunity and injustice," he'
                ' said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of'
                ' justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was'
                ' just the first step for the Palestinians. "As the Rome Statute today enters into force for the State'
                " of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a"
                ' State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she'
                ' said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize'
                " Palestine for joining the ICC should immediately end their pressure, and countries that support"
                " universal acceptance of the court's treaty should speak out to welcome its membership,\" said"
                " Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts"
                " to undermine international justice, not Palestine's decision to join a treaty to which over 100"
                ' countries around the world are members." In January, when the preliminary ICC examination was'
                " opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was"
                ' overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s'
                ' decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we'
                ' do not believe that it is eligible to join the ICC," the State Department said in a statement. It'
                ' urged the warring sides to resolve their differences through direct negotiations. "We will continue'
                ' to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said.'
                " But the ICC begs to differ with the definition of a state for its purposes and refers to the"
                ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows'
                " the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor"
                ' Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality."'
                " The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The"
                " inquiry will include alleged war crimes committed since June. The International Criminal Court was"
                " set up in 2002 to prosecute genocide, crimes against humanity and war crimes.",
                "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted"
                " Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor"
                ' Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A'
                " person who has such a video needs to immediately give it to the investigators.\" Robin's comments"
                " follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
                " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the"
                " French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was"
                " recovered from a phone at the wreckage site. The two publications described the supposed video, but"
                " did not post it on their websites. The publications said that they watched the video, which was"
                " found by a source close to the investigation. \"One can hear cries of 'My God' in several"
                ' languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps'
                " of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy"
                ' shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing'
                " scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident"
                " investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc"
                " Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the"
                ' Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell'
                ' phones have been collected at the site, he said, but that they "hadn\'t been exploited yet."'
                " Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute"
                " in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working"
                " hand-in-hand with investigators. But none of the cell phones found so far have been sent to the"
                " institute, Menichini said. Asked whether staff involved in the search could have leaked a memory"
                ' card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett:'
                ' Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are'
                ' "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
                ' cell phones from the crash site after Bild and Paris Match published their reports. "That is'
                " something we did not know before. ... Overall we can say many things of the investigation weren't"
                ' revealed by the investigation at the beginning," he said. What was mental state of Germanwings'
                " co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled"
                " depression years before he took the controls of Germanwings Flight 9525, which he's accused of"
                " deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school"
                ' in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email'
                " correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa"
                " said, included medical documents he submitted in connection with resuming his flight training. The"
                " announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle"
                " with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa,"
                " whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday"
                ' as a "swift and seamless clarification" and said it was sharing the information and documents --'
                " including training and medical records -- with public prosecutors. Spohr traveled to the crash site"
                " Wednesday, where recovery teams have been working for the past week to recover human remains and"
                " plane debris scattered across a steep mountainside. He saw the crisis center set up in"
                " Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving"
                " families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no"
                " visible human remains were left at the site but recovery teams would keep searching. French"
                " President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the"
                " victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
                " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini"
                " said. Among those personal belongings could be more cell phones belonging to the 144 passengers and"
                " six crew on board. Check out the latest from our correspondents . The details about Lubitz's"
                " correspondence with the flight school during his training were among several developments as"
                " investigators continued to delve into what caused the crash and Lubitz's possible motive for"
                " downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical"
                ' certificate, had passed all his examinations and "held all the licenses required." Earlier, a'
                " spokesman for the prosecutor's office in Dusseldorf, Christoph Kumpa, said medical records reveal"
                " Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent"
                " psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting"
                " Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether"
                " Lubitz feared his medical condition would cause him to lose his pilot's license, a European"
                ' government official briefed on the investigation told CNN on Tuesday. While flying was "a big part'
                " of his life,\" the source said, it's only one theory being considered. Another source, a law"
                " enforcement official briefed on the investigation, also told CNN that authorities believe the"
                " primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly"
                " because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor"
                " and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had"
                " psychological issues, the European government official said. But no matter what details emerge about"
                " his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
                ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the'
                " fact that maybe they weren't going to keep doing their job and they're upset about that and so"
                ' they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels'
                " entitled to also take that rage and turn it outward on 149 other people who had nothing to do with"
                " the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of"
                " Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from"
                " Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff,"
                " Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.",
            ],
            return_tensors="tf",
            padding="longest",
            truncation=True,
        )
        features = self.xsum_1_1_model.get_encoder()(**batch).last_hidden_state

        expected = np.array([[-0.0828, -0.0251, -0.0674], [0.1277, 0.3311, -0.0255], [0.2613, -0.0840, -0.2763]])
        assert np.allclose(features[0, :3, :3].numpy(), expected, atol=1e-3)
