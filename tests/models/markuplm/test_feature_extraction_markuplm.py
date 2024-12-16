# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import unittest

from transformers.testing_utils import require_bs4
from transformers.utils import is_bs4_available

from ...test_feature_extraction_common import FeatureExtractionSavingTestMixin


if is_bs4_available():
    from transformers import MarkupLMFeatureExtractor


class MarkupLMFeatureExtractionTester:
    def __init__(self, parent):
        self.parent = parent

    def prepare_feat_extract_dict(self):
        return {}


def get_html_strings():
    html_string_1 = """<HTML>

    <HEAD>
    <TITLE>sample document</TITLE>
    </HEAD>

    <BODY BGCOLOR="FFFFFF">
    <HR>
    <a href="http://google.com">Goog</a>
    <H1>This is one header</H1>
    <H2>This is a another Header</H2>
    <P>Travel from
        <P>
        <B>SFO to JFK</B>
        <BR>
        <B><I>on May 2, 2015 at 2:00 pm. For details go to confirm.com </I></B>
        <HR>
        <div style="color:#0000FF">
            <h3>Traveler <b> name </b> is
            <p> John Doe </p>
        </div>"""

    html_string_2 = """
    <!DOCTYPE html>
    <html>
    <body>

    <h1>My First Heading</h1>
    <p>My first paragraph.</p>

    </body>
    </html>
    """

    return [html_string_1, html_string_2]


@require_bs4
class MarkupLMFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):
    feature_extraction_class = MarkupLMFeatureExtractor if is_bs4_available() else None

    def setUp(self):
        self.feature_extract_tester = MarkupLMFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_call(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class()

        # Test not batched input
        html_string = get_html_strings()[0]
        encoding = feature_extractor(html_string)

        # fmt: off
        expected_nodes = [['sample document', 'Goog', 'This is one header', 'This is a another Header', 'Travel from', 'SFO to JFK', 'on May 2, 2015 at 2:00 pm. For details go to confirm.com', 'Traveler', 'name', 'is', 'John Doe']]
        expected_xpaths = [['/html/head/title', '/html/body/a', '/html/body/h1', '/html/body/h2', '/html/body/p', '/html/body/p/p/b[1]', '/html/body/p/p/b[2]/i', '/html/body/p/p/div/h3', '/html/body/p/p/div/h3/b', '/html/body/p/p/div/h3', '/html/body/p/p/div/h3/p']]
        # fmt: on

        self.assertEqual(encoding.nodes, expected_nodes)
        self.assertEqual(encoding.xpaths, expected_xpaths)

        # Test batched
        html_strings = get_html_strings()
        encoding = feature_extractor(html_strings)

        # fmt: off
        expected_nodes = expected_nodes + [['My First Heading', 'My first paragraph.']]
        expected_xpaths = expected_xpaths + [['/html/body/h1', '/html/body/p']]

        self.assertEqual(len(encoding.nodes), 2)
        self.assertEqual(len(encoding.xpaths), 2)

        self.assertEqual(encoding.nodes, expected_nodes)
        self.assertEqual(encoding.xpaths, expected_xpaths)
