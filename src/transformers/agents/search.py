#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import requests
from requests.exceptions import RequestException

from .tools import Tool


class DuckDuckGoSearchTool(Tool):
    name = "web_search"
    description = """Perform a web search based on your query (think a Google search) then returns the top search results as a list of dict elements.
    Each result has keys 'title', 'href' and 'body'."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "any"

    def forward(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
            )
        results = DDGS().text(query, max_results=7)
        return results


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given url and returns its content as a markdown string."
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            from markdownify import markdownify
        except ImportError:
            raise ImportError(
                "You must install package `markdownify` to run this tool: for instance run `pip install markdownify`."
            )
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return markdown_content

        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
