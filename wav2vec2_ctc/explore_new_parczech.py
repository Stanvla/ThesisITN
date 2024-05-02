# %%
import os
import pandas as pd
from tqdm import tqdm
import torchaudio
import matplotlib.pyplot as plt
import math


class Segment:
    def __init__(self, start, end, id, src_mp3_name, output_dir):
        self.start = start
        self.end = end
        self.id = id
        self.src_mp3_name = src_mp3_name
        self.output_dir = output_dir
        self.file_name = f'{self.id}.mp3'
        self.path = os.path.join(self.output_dir, self.src_mp3_name, self.file_name)

    def exists(self):
        return os.path.isfile(self.path)


def get_clean_df():
    dfs = []
    for i, f in tqdm(enumerate(os.listdir(asr_data_dir))):
        df = pd.read_feather(os.path.join(asr_data_dir, f))
        dfs.append(df)
    all_df = pd.concat(dfs)
    total_duration = all_df.duration.sum() / 3600
    good_df = all_df[(all_df.start_end_align == True) & (all_df.edit_distance < 1) & (all_df.cnt_speakers == 1) & (all_df.duration <= 20)]
    return good_df.copy(deep=True), total_duration


# %%
if __name__ == '__main__':
    # %%
    base_dir = '/lnet/express/work/people/stankov/alignment/new-parczech/baseline'
    asr_data_dir = os.path.join(base_dir, 'asr')
    align_data_dir = os.path.join(base_dir, 'align')

    good_df, total_duration = get_clean_df()
    good_duration = good_df.duration.sum() / 3600

    columns = [
        'mp3_name',
        'segment_idx',
        'global_start',
        'global_end',
        'duration',
        'speakers',
        'recognized_text'
    ]

    good_df = good_df[columns]
    good_df.speakers = good_df.speakers.astype(str)
    # %%
    good_df['avg_word_duration'] = good_df.duration / good_df.recognized_text.str.split().str.len()
    good_df = good_df[(good_df.avg_word_duration >= 0.22) & (good_df.avg_word_duration <= 0.76)]
    good_df = good_df[good_df.duration >= 0.5]

    good_df['n_words'] = good_df.recognized_text.str.split().str.len()
    good_df['n_chars'] = good_df.recognized_text.str.len()

    good_df = good_df[good_df.n_chars >= 7]
    good_df = good_df[good_df.n_words >= 1]

    # %%

    mp3_stat_df = good_df[['mp3_name', 'duration']].groupby('mp3_name').sum()
    mp3_stat_df.reset_index(inplace=True)
    mp3_stat_df['duration_minutes'] = mp3_stat_df.duration / 60
    reliable_mp3 = mp3_stat_df[mp3_stat_df.duration_minutes >= 0.5].mp3_name.values.tolist()

    reliable_df = good_df[good_df.mp3_name.isin(reliable_mp3)].copy(deep=True)

    reliable_duration = reliable_df.duration.sum() / 3600

    speakers_stat_df = good_df[['speakers', 'duration']].groupby('speakers').sum()
    speakers_stat_df.reset_index(inplace=True)
    speakers_stat_df['duration_minutes'] = speakers_stat_df.duration / 60
    test_speakers = speakers_stat_df[speakers_stat_df['duration_minutes'] <= 18].speakers.values.tolist()
    test_duration = speakers_stat_df[speakers_stat_df['duration_minutes'] <= 18].duration_minutes.sum() / 60

    reliable_df['split'] = 'train'
    reliable_df.loc[reliable_df.speakers.isin(test_speakers), 'split'] = 'test'

    print(f'total duration {total_duration}, reliable duration {reliable_duration}, test duration {test_duration}, reliable fraction {reliable_duration / total_duration}')
    print(reliable_df.shape)

    # read all source mp3 files
    # mp3_files[mp3_name] = path to mp3 file
    mp3_path = '/lnet/work/people/kopp/ParCzech/psp.cz/mp3/original_files/www.psp.cz/eknih'

    mp3_files = {}
    for root, subdir, files in os.walk(mp3_path):
        for f in files:
            mp3_files[f.replace('.mp3', '')] = os.path.join(root, f)

    output_dir = '/lnet/express/work/people/stankov/alignment/new-parczech/baseline/segments_asr'
    orig_sr = 48000
    new_sr = 16000
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr)

    for mp3 in tqdm(reliable_mp3):
        mp3_df = reliable_df[reliable_df.mp3_name == mp3][columns]
        segments = []
        for _, row in mp3_df.iterrows():
            _, segment_idx, start, end, *_ = row
            segment = Segment(start, end, segment_idx, mp3, output_dir)
            if not segment.exists():
                segments.append(segment)

        if segments != []:
            mp3_name = mp3_files[mp3]
            mp3_output_dir = os.path.join(output_dir, mp3)
            os.makedirs(mp3_output_dir, exist_ok=True)
            mp3_tensor, sr = torchaudio.load(mp3_name)
            for s in segments:
                start, end = math.floor(sr * s.start), math.ceil(sr * s.end)
                segment_tensor = mp3_tensor[:, start:end]
                resampled = resampler(segment_tensor)
                torchaudio.save(s.path, resampled, new_sr)



    # %%

