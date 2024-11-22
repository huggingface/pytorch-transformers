<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Testing


Let's take a look at how 🤗 Transformers models are tested and how you can write new tests and improve the existing ones.

There are 2 test suites in the repository:

1. `tests` -- tests for the general API
2. `examples` -- tests primarily for various applications that aren't part of the API

## How transformers are tested

1. Once a PR is submitted it gets tested with 9 CircleCi jobs. Every new commit to that PR gets retested. These jobs
   are defined in this [config file](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml), so that if needed you can reproduce the same
   environment on your machine.

   These CI jobs don't run `@slow` tests.

2. There are 3 jobs run by [github actions](https://github.com/huggingface/transformers/actions):

   - [torch hub integration](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml): checks whether torch hub
     integration works.

   - [self-hosted (push)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): runs fast tests on GPU only on commits on
     `main`. It only runs if a commit on `main` has updated the code in one of the following folders: `src`,
     `tests`, `.github` (to prevent running on added model cards, notebooks, etc.)

   - [self-hosted runner](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): runs normal and slow tests on GPU in
     `tests` and `examples`:

```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

   The results can be observed [here](https://github.com/huggingface/transformers/actions).



## Running tests



### Choosing which tests to run

This document goes into many details of how tests can be run. If after reading everything, you need even more details
you will find them [here](https://docs.pytest.org/en/latest/usage.html).

Here are some most useful ways of running tests.

Run all:

```console
pytest
```

or:

```bash
make test
```

Note that the latter is defined as:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

which tells pytest to:

- run as many test processes as they are CPU cores (which could be too many if you don't have a ton of RAM!)
- ensure that all tests from the same file will be run by the same test process
- do not capture output
- run in verbose mode



### Getting the list of all tests

All tests of the test suite:

```bash
pytest --collect-only -q
```

All tests of a given test file:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### Run a specific test module

To run an individual test module:

```bash
pytest tests/utils/test_logging.py
```

### Run specific tests

Since unittest is used inside most of the tests, to run specific subtests you need to know the name of the unittest
class containing those tests. For example, it could be:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

Here:

- `tests/test_optimization.py` - the file with tests
- `OptimizationTest` - the name of the class
- `test_adam_w` - the name of the specific test function

If the file contains multiple classes, you can choose to run only tests of a given class. For example:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

will run all the tests inside that class.

As mentioned earlier you can see what tests are contained inside the `OptimizationTest` class by running:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

You can run tests by keyword expressions.

To run only tests whose name contains `adam`:

```bash
pytest -k adam tests/test_optimization.py
```

Logical `and` and `or` can be used to indicate whether all keywords should match or either. `not` can be used to
negate.

To run all tests except those whose name contains `adam`:

```bash
pytest -k "not adam" tests/test_optimization.py
```

And you can combine the two patterns in one:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

For example to run both `test_adafactor` and `test_adam_w` you can use:

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

Note that we use `or` here, since we want either of the keywords to match to include both.

If you want to include only tests that include both patterns, `and` is to be used:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### Run `accelerate` tests

Sometimes you need to run `accelerate` tests on your models. For that you can just add `-m accelerate_tests` to your command, if let's say you want to run these tests on `OPT` run:

```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py
```


### Run documentation tests

In order to test whether the documentation examples are correct, you should check that the `doctests` are passing.
As an example, let's use [`WhisperModel.forward`'s docstring](https://github.com/huggingface/transformers/blob/1124d95dbb1a3512d3e80791d73d0f541d1d7e9f/src/transformers/models/whisper/modeling_whisper.py#L1591-L1609)

```python
r"""
Returns:

Example:
    ```python
    >>> import torch
    >>> from transformers import WhisperModel, WhisperFeatureExtractor
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```"""

```

Just run the following line to automatically test every docstring example in the desired file:
```bash
pytest --doctest-modules <path_to_file_or_dir>
```
If the file has a markdown extention, you should add the `--doctest-glob="*.md"` argument.

### Run only modified tests

You can run the tests related to the unstaged files or the current branch (according to Git) by using [pytest-picked](https://github.com/anapaulagomes/pytest-picked). This is a great way of quickly testing your changes didn't break
anything, since it won't run the tests related to files you didn't touch.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

All tests will be run from files and folders which are modified, but not yet committed.

### Automatically rerun failed tests on source modification

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) provides a very useful feature of detecting all failed
tests, and then waiting for you to modify files and continuously re-rerun those failing tests until they pass while you
fix them. So that you don't need to re start pytest after you made the fix. This is repeated until all tests pass after
which again a full run is performed.

```bash
pip install pytest-xdist
```

To enter the mode: `pytest -f` or `pytest --looponfail`

File changes are detected by looking at `looponfailroots` root directories and all of their contents (recursively).
If the default for this value does not work for you, you can change it in your project by setting a configuration
option in `setup.cfg`:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

or `pytest.ini`/``tox.ini`` files:

```ini
[pytest]
looponfailroots = transformers tests
```

This would lead to only looking for file changes in the respective directories, specified relatively to the ini-file’s
directory.

[pytest-watch](https://github.com/joeyespo/pytest-watch) is an alternative implementation of this functionality.


### Skip a test module

If you want to run all test modules, except a few you can exclude them by giving an explicit list of tests to run. For
example, to run all except `test_modeling_*.py` tests:

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### Clearing state

CI builds and when isolation is important (against speed), cache should be cleared:

```bash
pytest --cache-clear tests
```

### Running tests in parallel

As mentioned earlier `make test` runs tests in parallel via `pytest-xdist` plugin (`-n X` argument, e.g. `-n 2`
to run 2 parallel jobs).

`pytest-xdist`'s `--dist=` option allows one to control how the tests are grouped. `--dist=loadfile` puts the
tests located in one file onto the same process.

Since the order of executed tests is different and unpredictable, if running the test suite with `pytest-xdist`
produces failures (meaning we have some undetected coupled tests), use [pytest-replay](https://github.com/ESSS/pytest-replay) to replay the tests in the same order, which should help with then somehow
reducing that failing sequence to a minimum.

### Test order and repetition

It's good to repeat the tests several times, in sequence, randomly, or in sets, to detect any potential
inter-dependency and state-related bugs (tear down). And the straightforward multiple repetition is just good to detect
some problems that get uncovered by randomness of DL.


#### Repeat tests

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

And then run every test multiple times (50 by default):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>

This plugin doesn't work with `-n` flag from `pytest-xdist`.

</Tip>

<Tip>

There is another plugin `pytest-repeat`, but it doesn't work with `unittest`.

</Tip>

#### Run tests in a random order

```bash
pip install pytest-random-order
```

Important: the presence of `pytest-random-order` will automatically randomize tests, no configuration change or
command line options is required.

As explained earlier this allows detection of coupled tests - where one test's state affects the state of another. When
`pytest-random-order` is installed it will print the random seed it used for that session, e.g:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

So that if the given particular sequence fails, you can reproduce it by adding that exact seed, e.g.:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

It will only reproduce the exact order if you use the exact same list of tests (or no list at all). Once you start to
manually narrowing down the list you can no longer rely on the seed, but have to list them manually in the exact order
they failed and tell pytest to not randomize them instead using `--random-order-bucket=none`, e.g.:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

To disable the shuffling for all tests:

```bash
pytest --random-order-bucket=none
```

By default `--random-order-bucket=module` is implied, which will shuffle the files on the module levels. It can also
shuffle on `class`, `package`, `global` and `none` levels. For the complete details please see its
[documentation](https://github.com/jbasko/pytest-random-order).

Another randomization alternative is: [`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly). This
module has a very similar functionality/interface, but it doesn't have the bucket modes available in
`pytest-random-order`. It has the same problem of imposing itself once installed.

### Look and feel variations

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) is a plugin that improves the look-n-feel, adds a
progressbar, and show tests that fail and the assert instantly. It gets activated automatically upon installation.

```bash
pip install pytest-sugar
```

To run tests without it, run:

```bash
pytest -p no:sugar
```

or uninstall it.



#### Report each sub-test name and its progress

For a single or a group of tests via `pytest` (after `pip install pytest-pspec`):

```bash
pytest --pspec tests/test_optimization.py
```

#### Instantly shows failed tests

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) shows failures and errors instantly instead of
waiting until the end of test session.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### To GPU or not to GPU

On a GPU-enabled setup, to test in CPU-only mode add `CUDA_VISIBLE_DEVICES=""` for CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

or if you have multiple gpus, you can specify which one is to be used by `pytest`. For example, to use only the
second gpu if you have gpus `0` and `1`, you can run:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

For Intel GPUs, use `ZE_AFFINITY_MASK` instead of `CUDA_VISIBLE_DEVICES` in the above example.

This is handy when you want to run different tasks on different GPUs.

Some tests must be run on CPU-only, others on either CPU or GPU or TPU, yet others on multiple-GPUs. The following skip
decorators are used to set the requirements of tests CPU/GPU/XPU/TPU-wise:

- `require_torch` - this test will run only under torch
- `require_torch_gpu` - as `require_torch` plus requires at least 1 GPU
- `require_torch_multi_gpu` - as `require_torch` plus requires at least 2 GPUs
- `require_torch_non_multi_gpu` - as `require_torch` plus requires 0 or 1 GPUs
- `require_torch_up_to_2_gpus` - as `require_torch` plus requires 0 or 1 or 2 GPUs
- `require_torch_xla` - as `require_torch` plus requires at least 1 TPU

Let's depict the GPU requirements in the following table:


| n gpus | decorator                      |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


For example, here is a test that must be run only when there are 2 or more GPUs available and pytorch is installed:

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

If a test requires `tensorflow` use the `require_tf` decorator. For example:

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

These decorators can be stacked. For example, if a test is slow and requires at least one GPU under pytorch, here is
how to set it up:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

Some decorators like `@parametrized` rewrite test names, therefore `@require_*` skip decorators have to be listed
last for them to work correctly. Here is an example of the correct usage:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

This order problem doesn't exist with `@pytest.mark.parametrize`, you can put it first or last and it will still
work. But it only works with non-unittests.

Inside tests:

- How many GPUs are available:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count()  # works with torch and tf
```

### Testing with a specific PyTorch backend or device

To run the test suite on a specific torch device add `TRANSFORMERS_TEST_DEVICE="$device"` where `$device` is the target backend. For example, to test on CPU only:

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

This variable is useful for testing custom or less common PyTorch backends such as `mps`, `xpu` or `npu`. It can also be used to achieve the same effect as `CUDA_VISIBLE_DEVICES` by targeting specific GPUs or testing in CPU-only mode.

Certain devices will require an additional import after importing `torch` for the first time. This can be specified using the environment variable `TRANSFORMERS_TEST_BACKEND`:

```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```
Alternative backends may also require the replacement of device-specific functions. For example `torch.cuda.manual_seed` may need to be replaced with a device-specific seed setter like `torch.npu.manual_seed` or `torch.xpu.manual_seed` to correctly set a random seed on the device. To specify a new backend with backend-specific device functions when running the test suite, create a Python device specification file `spec.py` in the format:

```python
import torch
import torch_npu # for xpu, replace it with `import intel_extension_for_pytorch`
# !! Further additional imports can be added here !!

# Specify the device name (eg. 'cuda', 'cpu', 'npu', 'xpu', 'mps')
DEVICE_NAME = 'npu'

# Specify device-specific backends to dispatch to.
# If not specified, will fallback to 'default' in 'testing_utils.py`
MANUAL_SEED_FN = torch.npu.manual_seed
EMPTY_CACHE_FN = torch.npu.empty_cache
DEVICE_COUNT_FN = torch.npu.device_count
```
This format also allows for specification of any additional imports required. To use this file to replace equivalent methods in the test suite, set the environment variable `TRANSFORMERS_TEST_DEVICE_SPEC` to the path of the spec file, e.g. `TRANSFORMERS_TEST_DEVICE_SPEC=spec.py`.

Currently, only `MANUAL_SEED_FN`, `EMPTY_CACHE_FN` and `DEVICE_COUNT_FN` are supported for device-specific dispatch.

### Distributed training

`pytest` can't deal with distributed training directly. If this is attempted - the sub-processes don't do the right
thing and end up thinking they are `pytest` and start running the test suite in loops. It works, however, if one
spawns a normal process that then spawns off multiple workers and manages the IO pipes.

Here are some tests that use it:

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

To jump right into the execution point, search for the `execute_subprocess_async` call in those tests.

You will need at least 2 GPUs to see these tests in action:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### Output capture

During test execution any output sent to `stdout` and `stderr` is captured. If a test or a setup method fails, its
according captured output will usually be shown along with the failure traceback.

To disable output capturing and to get the `stdout` and `stderr` normally, use `-s` or `--capture=no`:

```bash
pytest -s tests/utils/test_logging.py
```

To send test results to JUnit format output:

```bash
pytest tests --junitxml=result.xml
```

### Color control

To have no color (e.g., yellow on white background is not readable):

```bash
pytest --color=no tests/utils/test_logging.py
```

### Sending test report to online pastebin service

Creating a URL for each test failure:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

This will submit test run information to a remote Paste service and provide a URL for each failure. You may select
tests as usual or add for example -x if you only want to send one particular failure.

Creating a URL for a whole test session log:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## Writing tests

🤗 transformers tests are based on `unittest`, but run by `pytest`, so most of the time features from both systems
can be used.

You can read [here](https://docs.pytest.org/en/stable/unittest.html) which features are supported, but the important
thing to remember is that most `pytest` fixtures don't work. Neither parametrization, but we use the module
`parameterized` that works in a similar way.


### Parametrization

Often, there is a need to run the same test multiple times, but with different arguments. It could be done from within
the test, but then there is no way of running that test for just one set of arguments.

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

Now, by default this test will be run 3 times, each time with the last 3 arguments of `test_floor` being assigned the
corresponding arguments in the parameter list.

and you could run just the `negative` and `integer` sets of params with:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

or all but `negative` sub-tests, with:

```bash
pytest -k "not negative" tests/test_mytest.py
```

Besides using the `-k` filter that was just mentioned, you can find out the exact name of each sub-test and run any
or all of them using their exact names.

```bash
pytest test_this1.py --collect-only -q
```

and it will list:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

So now you can run just 2 specific sub-tests:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

The module [parameterized](https://pypi.org/project/parameterized/) which is already in the developer dependencies
of `transformers` works for both: `unittests` and `pytest` tests.

If, however, the test is not a `unittest`, you may use `pytest.mark.parametrize` (or you may see it being used in
some existing tests, mostly under `examples`).

Here is the same example, this time using `pytest`'s `parametrize` marker:

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

Same as with `parameterized`, with `pytest.mark.parametrize` you can have a fine control over which sub-tests are
run, if the `-k` filter doesn't do the job. Except, this parametrization function creates a slightly different set of
names for the sub-tests. Here is what they look like:

```bash
pytest test_this2.py --collect-only -q
```

and it will list:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

So now you can run just the specific test:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

as in the previous example.



### Files and directories

In tests often we need to know where things are relative to the current test file, and it's not trivial since the test
could be invoked from more than one directory or could reside in sub-directories with different depths. A helper class
`transformers.test_utils.TestCasePlus` solves this problem by sorting out all the basic paths and provides easy
accessors to them:

- `pathlib` objects (all fully resolved):

  - `test_file_path` - the current test file path, i.e. `__file__`
  - `test_file_dir` - the directory containing the current test file
  - `tests_dir` - the directory of the `tests` test suite
  - `examples_dir` - the directory of the `examples` test suite
  - `repo_root_dir` - the directory of the repository
  - `src_dir` - the directory of `src` (i.e. where the `transformers` sub-dir resides)

- stringified paths---same as above but these return paths as strings, rather than `pathlib` objects:

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

To start using those all you need is to make sure that the test resides in a subclass of
`transformers.test_utils.TestCasePlus`. For example:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

If you don't need to manipulate paths via `pathlib` or you just need a path as a string, you can always invoked
`str()` on the `pathlib` object or use the accessors ending with `_str`. For example:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

### Temporary files and directories

Using unique temporary files and directories are essential for parallel test running, so that the tests won't overwrite
each other's data. Also we want to get the temporary files and directories removed at the end of each test that created
them. Therefore, using packages like `tempfile`, which address these needs is essential.

However, when debugging tests, you need to be able to see what goes into the temporary file or directory and you want
to know it's exact path and not having it randomized on every test re-run.

A helper class `transformers.test_utils.TestCasePlus` is best used for such purposes. It's a sub-class of
`unittest.TestCase`, so we can easily inherit from it in the test modules.

Here is an example of its usage:

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

This code creates a unique temporary directory, and sets `tmp_dir` to its location.

- Create a unique temporary dir:

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` will contain the path to the created temporary dir. It will be automatically removed at the end of the
test.

- Create a temporary dir of my choice, ensure it's empty before the test starts and don't empty it after the test.

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests didn't
leave any data in there.

- You can override the default behavior by directly overriding the `before` and `after` args, leading to one of the
  following behaviors:

  - `before=True`: the temporary dir will always be cleared at the beginning of the test.
  - `before=False`: if the temporary dir already existed, any existing files will remain there.
  - `after=True`: the temporary dir will always be deleted at the end of the test.
  - `after=False`: the temporary dir will always be left intact at the end of the test.

<Tip>

In order to run the equivalent of `rm -r` safely, only subdirs of the project repository checkout are allowed if
an explicit `tmp_dir` is used, so that by mistake no `/tmp` or similar important part of the filesystem will
get nuked. i.e. please always pass paths that start with `./`.

</Tip>

<Tip>

Each test can register multiple temporary directories and they all will get auto-removed, unless requested
otherwise.

</Tip>

### Temporary sys.path override

If you need to temporary override `sys.path` to import from another test for example, you can use the
`ExtendSysPath` context manager. Example:


```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### Skipping tests

This is useful when a bug is found and a new test is written, yet the bug is not fixed yet. In order to be able to
commit it to the main repository we need make sure it's skipped during `make test`.

Methods:

-  A **skip** means that you expect your test to pass only if some conditions are met, otherwise pytest should skip
  running the test altogether. Common examples are skipping windows-only tests on non-windows platforms, or skipping
  tests that depend on an external resource which is not available at the moment (for example a database).

-  A **xfail** means that you expect a test to fail for some reason. A common example is a test for a feature not yet
  implemented, or a bug not yet fixed. When a test passes despite being expected to fail (marked with
  pytest.mark.xfail), it’s an xpass and will be reported in the test summary.

One of the important differences between the two is that `skip` doesn't run the test, and `xfail` does. So if the
code that's buggy causes some bad state that will affect other tests, do not use `xfail`.

#### Implementation

- Here is how to skip whole test unconditionally:

```python no-style
@unittest.skip(reason="this bug needs to be fixed")
def test_feature_x():
```

or via pytest:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

or the `xfail` way:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```


Here's how to skip a test based on internal checks within the test:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

or the whole module:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

or the `xfail` way:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- Here is how to skip all tests in a module if some import is missing:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

-  Skip a test based on a condition:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

or:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

or skip the whole module:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

More details, example and ways are [here](https://docs.pytest.org/en/latest/skipping.html).

### Slow tests

The library of tests is ever-growing, and some of the tests take minutes to run, therefore we can't afford waiting for
an hour for the test suite to complete on CI. Therefore, with some exceptions for essential tests, slow tests should be
marked as in the example below:

```python no-style
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

Once a test is marked as `@slow`, to run such tests set `RUN_SLOW=1` env var, e.g.:

```bash
RUN_SLOW=1 pytest tests
```

Some decorators like `@parameterized` rewrite test names, therefore `@slow` and the rest of the skip decorators
`@require_*` have to be listed last for them to work correctly. Here is an example of the correct usage:

```python no-style
@parameterized.expand(...)
@slow
def test_integration_foo():
```

As explained at the beginning of this document, slow tests get to run on a scheduled basis, rather than in PRs CI
checks. So it's possible that some problems will be missed during a PR submission and get merged. Such problems will
get caught during the next scheduled CI job. But it also means that it's important to run the slow tests on your
machine before submitting the PR.

Here is a rough decision making mechanism for choosing which tests should be marked as slow:

If the test is focused on one of the library's internal components (e.g., modeling files, tokenization files,
pipelines), then we should run that test in the non-slow test suite. If it's focused on an other aspect of the library,
such as the documentation or the examples, then we should run these tests in the slow test suite. And then, to refine
this approach we should have exceptions:

- All tests that need to download a heavy set of weights or a dataset that is larger than ~50MB (e.g., model or
  tokenizer integration tests, pipeline integration tests) should be set to slow. If you're adding a new model, you
  should create and upload to the hub a tiny version of it (with random weights) for integration tests. This is
  discussed in the following paragraphs.
- All tests that need to do a training not specifically optimized to be fast should be set to slow.
- We can introduce exceptions if some of these should-be-non-slow tests are excruciatingly slow, and set them to
  `@slow`. Auto-modeling tests, which save and load large files to disk, are a good example of tests that are marked
  as `@slow`.
- If a test completes under 1 second on CI (including downloads if any) then it should be a normal test regardless.

Collectively, all the non-slow tests need to cover entirely the different internals, while remaining fast. For example,
a significant coverage can be achieved by testing with specially created tiny models with random weights. Such models
have the very minimal number of layers (e.g., 2), vocab size (e.g., 1000), etc. Then the `@slow` tests can use large
slow models to do qualitative testing. To see the use of these simply look for *tiny* models with:

```bash
grep tiny tests examples
```

Here is an example of a [script](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py) that created the tiny model
[stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de). You can easily adjust it to your specific
model's architecture.

It's easy to measure the run-time incorrectly if for example there is an overheard of downloading a huge model, but if
you test it locally the downloaded files would be cached and thus the download time not measured. Hence check the
execution speed report in CI logs instead (the output of `pytest --durations=0 tests`).

That report is also useful to find slow outliers that aren't marked as such, or which need to be re-written to be fast.
If you notice that the test suite starts getting slow on CI, the top listing of this report will show the slowest
tests.


### Testing the stdout/stderr output

In order to test functions that write to `stdout` and/or `stderr`, the test can access those streams using the
`pytest`'s [capsys system](https://docs.pytest.org/en/latest/capture.html). Here is how this is accomplished:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```

And, of course, most of the time, `stderr` will come as a part of an exception, so try/except has to be used in such
a case:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

Another approach to capturing stdout is via `contextlib.redirect_stdout`:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

An important potential issue with capturing stdout is that it may contain `\r` characters that in normal `print`
reset everything that has been printed so far. There is no problem with `pytest`, but with `pytest -s` these
characters get included in the buffer, so to be able to have the test run with and without `-s`, you have to make an
extra cleanup to the captured output, using `re.sub(r'~.*\r', '', buf, 0, re.M)`.

But, then we have a helper context manager wrapper to automatically take care of it all, regardless of whether it has
some `\r`'s in it or not, so it's a simple:

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

Here is a full test example:

```python
from transformers.testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

If you'd like to capture `stderr` use the `CaptureStderr` class instead:

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

If you need to capture both streams at once, use the parent `CaptureStd` class:

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

Also, to aid debugging test issues, by default these context managers automatically replay the captured streams on exit
from the context.


### Capturing logger stream

If you need to validate the output of a logger, you can use `CaptureLogger`:

```python
from transformers import logging
from transformers.testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### Testing with environment variables

If you want to test the impact of environment variables for a specific test you can use a helper decorator
`transformers.testing_utils.mockenv`

```python
from transformers.testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

At times an external program needs to be called, which requires setting `PYTHONPATH` in `os.environ` to include
multiple local paths. A helper class `transformers.test_utils.TestCasePlus` comes to help:

```python
from transformers.testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

Depending on whether the test file was under the `tests` test suite or `examples` it'll correctly set up
`env[PYTHONPATH]` to include one of these two directories, and also the `src` directory to ensure the testing is
done against the current repo, and finally with whatever `env[PYTHONPATH]` was already set to before the test was
called if anything.

This helper method creates a copy of the `os.environ` object, so the original remains intact.


### Getting reproducible results

In some situations you may want to remove randomness for your tests. To get identical reproducible results set, you
will need to fix the seed:

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
import tensorflow as tf 

tf.random.set_seed(seed)
```

### Debugging tests

To start a debugger at the point of the warning, do this:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## Working with github actions workflows

To trigger a self-push workflow CI job, you must:

1. Create a new branch on `transformers` origin (not a fork!).
2. The branch name has to start with either `ci_` or `ci-` (`main` triggers it too, but we can't do PRs on
   `main`). It also gets triggered only for specific paths - you can find the up-to-date definition in case it
   changed since this document has been written [here](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml) under *push:*
3. Create a PR from this branch.
4. Then you can see the job appear [here](https://github.com/huggingface/transformers/actions/workflows/self-push.yml). It may not run right away if there
   is a backlog.




## Testing Experimental CI Features

Testing CI features can be potentially problematic as it can interfere with the normal CI functioning. Therefore if a
new CI feature is to be added, it should be done as following.

1. Create a new dedicated job that tests what needs to be tested
2. The new job must always succeed so that it gives us a green ✓ (details below).
3. Let it run for some days to see that a variety of different PR types get to run on it (user fork branches,
   non-forked branches, branches originating from github.com UI direct file edit, various forced pushes, etc. - there
   are so many) while monitoring the experimental job's logs (not the overall job green as it's purposefully always
   green)
4. When it's clear that everything is solid, then merge the new changes into existing jobs.

That way experiments on CI functionality itself won't interfere with the normal workflow.

Now how can we make the job always succeed while the new CI feature is being developed?

Some CIs, like TravisCI support ignore-step-failure and will report the overall job as successful, but CircleCI and
Github Actions as of this writing don't support that.

So the following workaround can be used:

1. `set +euo pipefail` at the beginning of the run command to suppress most potential failures in the bash script.
2. the last command must be a success: `echo "done"` or just `true` will do

Here is an example:

```yaml
- run:
    name: run CI experiment
    command: |
        set +euo pipefail
        echo "setting run-all-despite-any-errors-mode"
        this_command_will_fail
        echo "but bash continues to run"
        # emulate another failure
        false
        # but the last command must be a success
        echo "during experiment do not remove: reporting success to CI, even if there were failures"
```

For simple commands you could also do:

```bash
cmd_that_may_fail || true
```

Of course, once satisfied with the results, integrate the experimental step or job with the rest of the normal jobs,
while removing `set +euo pipefail` or any other things you may have added to ensure that the experimental job doesn't
interfere with the normal CI functioning.

This whole process would have been much easier if we only could set something like `allow-failure` for the
experimental step, and let it fail without impacting the overall status of PRs. But as mentioned earlier CircleCI and
Github Actions don't support it at the moment.

You can vote for this feature and see where it is at these CI-specific threads:

- [Github Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)

## DeepSpeed integration

For a PR that involves the DeepSpeed integration, keep in mind our CircleCI PR CI setup doesn't have GPUs. Tests requiring GPUs are run on a different CI nightly. This means if you get a passing CI report in your PR, it doesn’t mean the DeepSpeed tests pass.

To run DeepSpeed tests:

```bash
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

Any changes to the modeling or PyTorch examples code requires running the model zoo tests as well.

```bash
RUN_SLOW=1 pytest tests/deepspeed
```
