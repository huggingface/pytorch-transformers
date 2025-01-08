# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for MarkupLM.
"""

import html

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends


if is_bs4_available():
    import bs4
    from bs4 import BeautifulSoup


logger = logging.get_logger(__name__)


class MarkupLMFeatureExtractor(FeatureExtractionMixin):
    r"""
    Constructs a MarkupLM feature extractor. This can be used to get a list of nodes and corresponding xpaths from HTML
    strings.

    This feature extractor inherits from [`~feature_extraction_utils.PreTrainedFeatureExtractor`] which contains most
    of the main methods. Users should refer to this superclass for more information regarding those methods.

    """

    def __init__(self, **kwargs):
        requires_backends(self, ["bs4"])
        super().__init__(**kwargs)

    def xpath_soup(self, element):
        xpath_tags = []
        xpath_subscripts = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child)
            )
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        return xpath_tags, xpath_subscripts

    def get_three_from_single(self, html_string):
        html_code = BeautifulSoup(html_string, "html.parser")

        all_doc_strings = []
        string2xtag_seq = []
        string2xsubs_seq = []

        for element in html_code.descendants:
            if isinstance(element, bs4.element.NavigableString):
                if type(element.parent) is not bs4.element.Tag:
                    continue

                text_in_this_tag = html.unescape(element).strip()
                if not text_in_this_tag:
                    continue

                all_doc_strings.append(text_in_this_tag)

                xpath_tags, xpath_subscripts = self.xpath_soup(element)
                string2xtag_seq.append(xpath_tags)
                string2xsubs_seq.append(xpath_subscripts)

        if len(all_doc_strings) != len(string2xtag_seq):
            raise ValueError("Number of doc strings and xtags does not correspond")
        if len(all_doc_strings) != len(string2xsubs_seq):
            raise ValueError("Number of doc strings and xsubs does not correspond")

        return all_doc_strings, string2xtag_seq, string2xsubs_seq

    def construct_xpath(self, xpath_tags, xpath_subscripts):
        xpath = ""
        for tagname, subs in zip(xpath_tags, xpath_subscripts):
            xpath += f"/{tagname}"
            if subs != 0:
                xpath += f"[{subs}]"
        return xpath

    def __call__(self, html_strings) -> BatchFeature:
        """
        Main method to prepare for the model one or several HTML strings.

        Args:
            html_strings (`str`, `List[str]`):
                The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **nodes** -- Nodes.
            - **xpaths** -- Corresponding xpaths.

        Examples:

        ```python
        >>> from transformers import MarkupLMFeatureExtractor

        >>> page_name_1 = "page1.html"
        >>> page_name_2 = "page2.html"
        >>> page_name_3 = "page3.html"

        >>> with open(page_name_1) as f:
        ...     single_html_string = f.read()

        >>> feature_extractor = MarkupLMFeatureExtractor()

        >>> # single example
        >>> encoding = feature_extractor(single_html_string)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])

        >>> # batched example

        >>> multi_html_strings = []

        >>> with open(page_name_2) as f:
        ...     multi_html_strings.append(f.read())
        >>> with open(page_name_3) as f:
        ...     multi_html_strings.append(f.read())

        >>> encoding = feature_extractor(multi_html_strings)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])
        ```"""

        # Input type checking for clearer error
        valid_strings = False

        # Check that strings has a valid type
        if isinstance(html_strings, str):
            valid_strings = True
        elif isinstance(html_strings, (list, tuple)):
            if len(html_strings) == 0 or isinstance(html_strings[0], str):
                valid_strings = True

        if not valid_strings:
            raise ValueError(
                "HTML strings must of type `str`, `List[str]` (batch of examples), "
                f"but is of type {type(html_strings)}."
            )

        is_batched = bool(isinstance(html_strings, (list, tuple)) and (isinstance(html_strings[0], str)))

        if not is_batched:
            html_strings = [html_strings]

        # Get nodes + xpaths
        nodes = []
        xpaths = []
        for html_string in html_strings:
            all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)
            nodes.append(all_doc_strings)
            xpath_strings = []
            for node, tag_list, sub_list in zip(all_doc_strings, string2xtag_seq, string2xsubs_seq):
                xpath_string = self.construct_xpath(tag_list, sub_list)
                xpath_strings.append(xpath_string)
            xpaths.append(xpath_strings)

        # return as Dict
        data = {"nodes": nodes, "xpaths": xpaths}
        encoded_inputs = BatchFeature(data=data, tensor_type=None)

        return encoded_inputs


__all__ = ["MarkupLMFeatureExtractor"]
