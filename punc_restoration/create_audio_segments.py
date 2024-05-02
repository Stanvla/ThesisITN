# %%
import os
import sys

import pandas as pd
from tqdm import tqdm
import torchaudio
import matplotlib.pyplot as plt
import math
import multiprocessing
import logging

try:
    from args import get_args
except ModuleNotFoundError as e:
    from punc_restoration.args import get_args
# %%

class Segment:
    def __init__(self, start, end, id, src_mp3_name):
        self.start = start
        self.end = end
        self.id = id
        self.src_mp3_name = src_mp3_name

        self.file_name = f'{self.id}.mp3'
        self.output_dir = os.path.join(params['data_dir'], 'data')
        self.path = os.path.join(self.output_dir, self.src_mp3_name, self.file_name)

    def exists(self):
        return os.path.isfile(self.path)


def main(inputs):
    idx, resampler, new_sr, mp3, first_pid = inputs

    # for mp3 in tqdm(all_mp3s):
    mp3_df =main_df[main_df.mp3 == mp3]

    segments = []
    for _, row in mp3_df.iterrows():
        segment_idx, _, start, end, *_ = row
        segment = Segment(start, end, segment_idx, mp3)

        if segment.exists():
            continue
        segments.append(segment)

    if segments != []:

        # if multiprocessing.current_process().pid == first_pid:
        # if idx % 10 == 0:
        logging.debug(f'{idx / len(all_mp3s):.4f} :: {idx:4} / {len(all_mp3s)} :: {multiprocessing.current_process().pid}')

        mp3_output_dir = segments[0].output_dir
        os.makedirs(mp3_output_dir, exist_ok=True)

        src_mp3_path = mp3_files[mp3]
        mp3_tensor, sr = torchaudio.load(src_mp3_path)
        mp3_tensor = resampler(mp3_tensor)

        for s in segments:
            start, end = math.floor(new_sr * s.start), math.ceil(new_sr * s.end)
            segment_tensor = mp3_tensor[:, start:end]
            # resampled = resampler(segment_tensor)
            # torchaudio.save(s.path, segment_tensor, new_sr)

# %%
if __name__ == '__main__':
    # %%

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(message)s',
                        datefmt='%H:%M:%S %d.%m.%Y')

    logging.debug('Start')

    params = get_args()

    mp3_files = {}
    for root, subdir, files in os.walk(params['mp3_path']):
        for f in files:
            mp3_files[f.replace('.mp3', '')] = os.path.join(root, f)


    # %%
    main_df = pd.read_parquet(
        os.path.join(params['data_dir'], params['main_df'])
    )
    # %%
    orig_sr = 48000
    new_sr = 16000
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)

    # %%
    all_mp3s = main_df.mp3.unique()
    def gen_input():
        for i, mp3 in enumerate(tqdm(all_mp3s)):
        # for i, mp3 in enumerate(all_mp3s):
            yield [i, resampler, new_sr, mp3, 0]

    for inp in gen_input():
        main(inp)

    # with multiprocessing.Pool(8) as pool:
    #     first_pid = multiprocessing.active_children()[1].pid
    #
    #     def gen_input():
    #         # for i, mp3 in enumerate(tqdm(all_mp3s)):
    #         for i, mp3 in enumerate(all_mp3s):
    #             yield [i, resampler, new_sr, mp3, first_pid]
    #
    #
    #     pool.map(main, gen_input())
    # %%


    # %%
    # %%
    # %%
    # %%
    # %%
    # %%