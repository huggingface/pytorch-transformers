# Copyright 2023 The HuggingFace Team. All rights reserved.
import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union

import numpy as np


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
    ffmpeg_input_device: Optional[str] = None,
    ffmpeg_additional_args: Optional[list[str]] = None,
):
    """
    Helper function to read audio from a microphone using ffmpeg. The default input device will be used unless another
    input device is specified using the `ffmpeg_input_device` argument. Uses 'alsa' on Linux, 'avfoundation' on MacOS and
    'dshow' on Windows.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned.
        format_for_conversion (`str`, defaults to `f32le`):
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
        ffmpeg_input_device (`str`, *optional*):
            The identifier of the input device to be used by ffmpeg (i.e. ffmpeg's '-i' argument). If unset,
            the default input device will be used. See `https://www.ffmpeg.org/ffmpeg-devices.html#Input-Devices`
            for how to specify and list input devices.
        ffmpeg_additional_args (`list[str]`, *optional*):
            Additional arguments to pass to ffmpeg, can include arguments like -nostdin for running as a background
            process. For example, to pass -nostdin to the ffmpeg process, pass in ["-nostdin"]. If passing in flags
            with multiple arguments, use the following convention (eg ["flag", "arg1", "arg2]).

    Returns:
        A generator yielding audio chunks of `chunk_length_s` seconds as `bytes` objects of length
        `int(round(sampling_rate * chunk_length_s)) * size_of_sample`.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    if format_for_conversion == "s16le":
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()

    if system == "Linux":
        format_ = "alsa"
        input_ = ffmpeg_input_device or "default"
    elif system == "Darwin":
        format_ = "avfoundation"
        input_ = ffmpeg_input_device or ":default"
    elif system == "Windows":
        format_ = "dshow"
        input_ = ffmpeg_input_device or _get_microphone_name()

    ffmpeg_additional_args = [] if ffmpeg_additional_args is None else ffmpeg_additional_args

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        input_,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    ffmpeg_command.extend(ffmpeg_additional_args)

    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    for item in iterator:
        yield item


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
    ffmpeg_input_device: Optional[str] = None,
    ffmpeg_additional_args: Optional[list[str]] = None,
):
    """
    Helper function to read audio from a microphone using ffmpeg. This will output `partial` overlapping chunks starting
    from `stream_chunk_s` (if it is defined) until `chunk_length_s` is reached. It will make use of striding to avoid
    errors on the "sides" of the various chunks. The default input device will be used unless another input device is
    specified using the `ffmpeg_input_device` argument. Uses 'alsa' on Linux, 'avfoundation' on MacOS and 'dshow' on Windows.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned. This includes the eventual striding.
        stream_chunk_s (`float` or `int`):
            The length of the minimal temporary audio to be returned.
        stride_length_s (`float` or `int` or `(float, float)`, *optional*):
            The length of the striding to be used. Stride is used to provide context to a model on the (left, right) of
            an audio sample but without using that part to actually make the prediction. Setting this does not change
            the length of the chunk.
        format_for_conversion (`str`, *optional*, defaults to `f32le`):
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
        ffmpeg_input_device (`str`, *optional*):
            The identifier of the input device to be used by ffmpeg (i.e. ffmpeg's '-i' argument). If unset,
            the default input device will be used. See `https://www.ffmpeg.org/ffmpeg-devices.html#Input-Devices`
            for how to specify and list input devices.
        ffmpeg_additional_args (`list[str]`, *optional*):
            Additional arguments to pass to ffmpeg, can include arguments like -nostdin for running as a background
            process. For example, to pass -nostdin to the ffmpeg process, pass in ["-nostdin"]. If passing in flags
            with multiple arguments, use the following convention (eg ["flag", "arg1", "arg2]).

    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.array(), "partial" bool}` With optionally a `"stride" (int, int)` key if
        `stride_length_s` is defined.

        `stride` and `raw` are all expressed in `samples`, and `partial` is a boolean saying if the current yield item
        is a whole chunk, or a partial temporary result to be later replaced by another larger chunk.
    """
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    microphone = ffmpeg_microphone(
        sampling_rate,
        chunk_s,
        format_for_conversion=format_for_conversion,
        ffmpeg_input_device=ffmpeg_input_device,
        ffmpeg_additional_args=[] if ffmpeg_additional_args is None else ffmpeg_additional_args,
    )

    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # Put everything back in numpy scale
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        item["sampling_rate"] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            # We're late !! SKIP
            continue
        yield item


def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                # We are flushing the accumulator
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right :]
    # Last chunk
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item


def _ffmpeg_stream(ffmpeg_command, buflen: int):
    """
    Internal function to create the generator of data through ffmpeg
    """
    bufsize = 2**24  # 16Mo
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error


def _get_microphone_name():
    """
    Retrieve the microphone name in Windows .
    """
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]

        if microphone_lines:
            microphone_name = microphone_lines[0].split('"')[1]
            print(f"Using microphone: {microphone_name}")
            return f"audio={microphone_name}"
    except FileNotFoundError:
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")

    return "default"
