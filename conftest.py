# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import doctest
import sys
import warnings
from os.path import abspath, dirname, join

import pytest  # Removed unused import of _pytest

from transformers.testing_utils import HfDoctestModule, HfDocTestParser

# Tests that are not device-specific
NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_processor",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_save_load_fast_init_from_base",
    "test_fast_init_context_manager",
    "test_fast_init_tied_embeddings",
    "test_save_load_fast_init_to_base",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_pt_tf_model_equivalence",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
    "/agents/",
}

# Add the "src" directory to sys.path to ensure modules are discoverable
repo_src_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, repo_src_path)  # Renamed for clarity

# Suppress FutureWarning globally to avoid cluttering test logs
warnings.simplefilter(action="ignore", category=FutureWarning)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )
    config.addinivalue_line(
        "markers", "is_pt_flax_cross_test: mark test to run only when PT and FLAX interactions are tested"
    )
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "agent_tests: mark the agent tests that are run on their specific schedule")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")

def pytest_collection_modifyitems(items):
    """Modify test collection to add markers for NOT_DEVICE_TESTS."""
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)

def pytest_addoption(parser):
    """Add shared pytest options."""
    from transformers.testing_utils import pytest_addoption_shared
    pytest_addoption_shared(parser)

def pytest_terminal_summary(terminalreporter):
    """Generate custom test summary reports."""
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)

def pytest_sessionfinish(session, exitstatus):
    """Ensure exit status doesn't fail CI when no tests are collected."""
    if exitstatus == 5:  # Exit status 5 indicates no tests collected
        session.exitstatus = 0

# Register custom doctest flag to ignore result output
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

class CustomOutputChecker(doctest.OutputChecker):
    """Custom output checker to handle IGNORE_RESULT flag."""
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return super().check_output(want, got, optionflags)

doctest.OutputChecker = CustomOutputChecker
pytest.DoctestModule = HfDoctestModule  # Corrected reference for pytest doctest module
doctest.DocTestParser = HfDocTestParser
