import hashlib
import os
from typing import Iterable
import shutil
import subprocess
import re
import time
from typing import Union
import functools
import torch
from torch import Tensor
from collections.abc import Mapping

from .logger import logger
import folder_paths

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)
DIMMAX = 8192
ENCODE_ARGS = ("utf-8", 'backslashreplace')

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3),
                        ("svtav1", 5), ("libopus", 1)]
    for criterion in simple_criterion:
        if criterion[0] in version:
            score += criterion[1]
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types

imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

class ContainsAll(dict):
    def __contains__(self, other):
        return True
    def __getitem__(self, key):
        return super().get(key, (None, {}))

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

gifski_path = os.environ.get("VHS_GIFSKI", None) or os.environ.get("JOV_GIFSKI", None) or shutil.which("gifski")
ytdl_path = os.environ.get("VHS_YTDL", None) or shutil.which('yt-dlp') or shutil.which('youtube-dl')

download_history = {}

def try_download_video(url):
    if ytdl_path is None:
        return None
    if url in download_history:
        return download_history[url]
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    try:
        res = subprocess.run([ytdl_path, "--print", "after_move:filepath",
                              "-P", folder_paths.get_temp_directory(), url],
                             capture_output=True, check=True)
        file = res.stdout.decode(*ENCODE_ARGS).strip()
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the yt-dl process:\n" +
                        e.stderr.decode(*ENCODE_ARGS))
    download_history[url] = file
    return file

def is_safe_path(path, strict=False):
    if "VHS_STRICT_PATHS" not in os.environ and not strict:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        return False
    return common_path == basedir

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int = 0, select_every_nth: int = 1, extensions: Iterable = None):
    directory = strip_path(directory)
    dir_files = sorted([os.path.join(directory, x) for x in os.listdir(directory)
                       if os.path.isfile(os.path.join(directory, x))])
    if extensions is not None:
        extensions = list(extensions)
        dir_files = [f for f in dir_files if "." + f.split(".")[-1].lower() in extensions]
    return dir_files[skip_first_images::select_every_nth]

def calculate_file_hash(filename: str, hash_every_n: int = 1):
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"], capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ', res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" + e.stderr.decode(*ENCODE_ARGS))
    ar = 44100
    ac = 2
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._dict = None
    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]
    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)
    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)

def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)

def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_sequence(path):
    (path, file) = os.path.split(path)
    if not os.path.isdir(path):
        return False
    match = re.search('%0?\\d+d', file)
    if not match:
        return False
    seq = '\\\\d+'
    if match.group() != '%d':
        seq = f'\\\\d{{{match.group()[1:-1]}}}'
    file_matcher = re.compile(re.sub('%0?\\d+d', seq, file))
    return any(file_matcher.fullmatch(f) for f in os.listdir(path))

def strip_path(path):
    path = path.strip()
    return path.strip('"')

def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(strip_path(path))

def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return f"Invalid file path: {path}"
    return is_safe_path(path)

def validate_index(index: int, length: int = 0, is_range: bool = False, allow_negative: bool = False, allow_missing: bool = False) -> int:
    if is_range:
        return index
    if length > 0 and index > length-1 and not allow_missing:
        raise IndexError(f"Index '{index}' out of range for {length} item(s).")
    if index < 0:
        if not allow_negative:
            raise IndexError(f"Negative indices not allowed, but was '{index}'.")
        conv_index = length + index
        if conv_index < 0 and not allow_missing:
            raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for {length} item(s).")
        index = conv_index
    return index

def convert_to_index_int(raw_index: str, length: int = 0, is_range: bool = False, allow_negative: bool = False, allow_missing: bool = False) -> int:
    try:
        return validate_index(int(raw_index), length=length, is_range=is_range, allow_negative=allow_negative, allow_missing=allow_missing)
    except ValueError as e:
        raise ValueError(f"Index '{raw_index}' must be an integer.", e)

def convert_str_to_indexes(indexes_str: str, length: int = 0, allow_missing=False) -> list[int]:
    if not indexes_str:
        return []
    int_indexes = list(range(0, length))
    allow_negative = length > 0
    chosen_indexes = []
    for g in map(str.strip, indexes_str.split(",")):
        if ':' in g:
            parts = list(map(str.strip, g.split(":", 2)))
            start = convert_to_index_int(parts[0], length, True, allow_negative, allow_missing) if parts[0] else 0
            end = convert_to_index_int(parts[1], length, True, allow_negative, allow_missing) if parts[1] else length
            step = convert_to_index_int(parts[2], length, True, True, True) if len(parts) > 2 and parts[2] else 1
            chosen_indexes.extend(int_indexes[start:end][::step])
        else:
            chosen_indexes.append(convert_to_index_int(g, length, allow_negative, allow_missing))
    return chosen_indexes

def select_indexes(input_obj: Union[Tensor, list], idxs: list):
    return input_obj[idxs] if isinstance(input_obj, Tensor) else [input_obj[i] for i in idxs]

def merge_filter_args(args, ftype="-vf"):
    try:
        start_index = args.index(ftype)+1
        index = start_index
        while True:
            index = args.index(ftype, index)
            args[start_index] += ',' + args[index+1]
            args.pop(index)
            args.pop(index)
    except ValueError:
        pass

def select_indexes_from_str(input_obj: Union[Tensor, list], indexes: str, err_if_missing=True, err_if_empty=True):
    real_idxs = convert_str_to_indexes(indexes, len(input_obj), allow_missing=not err_if_missing)
    if err_if_empty and len(real_idxs) == 0:
        raise Exception(f"Nothing was selected based on indexes found in '{indexes}'.")
    return select_indexes(input_obj, real_idxs)

def hook(obj, attr):
    def dec(f):
        f = functools.update_wrapper(f, getattr(obj, attr))
        setattr(obj, attr, f)
        return f
    return dec

def cached(duration):
    def dec(f):
        cached_ret = None
        cache_time = 0
        def cached_func():
            nonlocal cache_time, cached_ret
            if time.time() > cache_time + duration or cached_ret is None:
                cache_time = time.time()
                cached_ret = f()
            return cached_ret
        return cached_func
    return dec