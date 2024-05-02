# %%
import os
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from collections import Counter
import warnings


def display_avg_dur(lb, ub):
    avg_char_durations = []
    for f in tqdm(os.listdir(align_data_dir)):
        df = pd.read_feather(os.path.join(align_data_dir, f))
        df = df[df.dist == '0']
        df['avg_char_dur'] = df.end - df.start
        df['avg_char_dur'] = df['avg_char_dur'] / df.rec_word.str.len()
        avg_char_durations.extend(df.avg_char_dur.tolist())

    filtered_avg_char_durations = [x for x in avg_char_durations if lb < x < ub]
    print(len(filtered_avg_char_durations) / len(avg_char_durations))
    filtered_avg_char_durations = sorted(filtered_avg_char_durations)
    plt.plot(range(len(filtered_avg_char_durations)), filtered_avg_char_durations)
    plt.show()


def get_punc_segments(df, min_len, min_dur):
    # return list of punc_segments
    # each segment contains pair of aligned words
    segments = []
    segment = []
    prev_word_id = -1

    for row in df.itertuples():
        if row.gold_word == blank or row.rec_word == blank:
            segments.append(segment)
            segment = []
            continue

        curr_word_id = row.Index
        avg_word_duration = (row.end - row.start) / len(row.rec_word)
        # check if word was recognized "correctly"
        if row.dist == '0' and (avg_char_dur_lb < avg_word_duration < avg_char_dur_ub):
            # check if there were any skips
            if curr_word_id == prev_word_id + 1:
                segment.append(row)
            else:
                segments.append(segment)
                segment = [row]
        else:
            segments.append(segment)
            segment = []
        prev_word_id = curr_word_id

    return [s for s in segments if len(s) > min_len and s[-1].end - s[0].start > min_dur]


# def split_long_segment(s, min_offset=3, max_offset=20):
def split_long_segment(s, p, n):
    short_segments = []
    short_segment = []
    offset = np.random.binomial(n, p, 1)[0]
    useful_flag = False
    # print(offset)
    for w in s:
        if useful_flag:
            offset -= 1

        useful_flag = useful_flag or w.useful
        short_segment.append(w)

        if offset == 0:
            short_segments.append(short_segment)
            # display_segment(short_segment)
            short_segment = []
            offset = np.random.binomial(n, p, 1)[0]
            useful_flag = False
            # print(offset)

    # last short segment may be missed
    if short_segment != []:
        # last segment may not contain punc
        if any(w.useful for w in short_segment):
            short_segments.append(short_segment)
        elif short_segments != []:
            short_segments[-1].extend(short_segment)

        # display_segment(short_segments[-1])
    return short_segments


def segment_get_speakers(s):
    return set([row.speaker for row in s])


def display_segment(s):
    start =s[0].start
    end = s[-1].end
    for w in s:
        print('{index:4} {segment_index:3}    {rec_word:25} {gold_word:25} {speaker:35} {label:<15} {start:3.3f}   {end:3.3f}   {avg_dur:.4f}'.format(
            index=w.Index,
            segment_index=w.segment_idx,
            rec_word=w.rec_word,
            gold_word=w.gold_word,
            speaker=w.speaker,
            label=w.punc,
            start=w.start,
            end=w.end,
            avg_dur=(w.end - w.start) / len(w.rec_word),
        ))
    print(f'duration :: {end - start:.3f}')
    print('-----' * 25)


def split_by_speaker(s):
    new_segments = []
    new_segment = []
    speaker = s[0].speaker
    for r in s:
        if r.speaker != speaker:
            new_segments.append(new_segment)
            new_segment = []
        new_segment.append(r)
        speaker = r.speaker
    if new_segment != []:
        new_segments.append(new_segment)
    return new_segments


def dataset_statistics(files, zero_edit_dist):
    statistics = {
        'speakers': [],
        'words': 0,
        'duration': 0,
        'files': len(files)
    }
    for f in files:
        df = read_df(f)

        non_zero_duration = 0
        if zero_edit_dist:
            non_zero_duration = np.sum(df.loc[df.dist != '0', 'end'].values - df.loc[df.dist != '0', 'start'].values)
            df = df[df.dist == '0'].copy()
            if len(df) == 0:
                continue

        statistics['duration'] += (df.iloc[-1].end - df.iloc[0].start) - non_zero_duration
        statistics['words'] += len(df[df.gold_word != '<blank>'])
        statistics['speakers'].extend(df.speaker.unique())
    statistics['speakers'] = len(set(statistics['speakers']))
    return statistics


def get_segments(file):
    # return list of punc_segments
    # each segment contains pair of aligned words
    df = read_df(file)
    df = df[(df.dist == '0') & (df.gold_word != blank) & (df.rec_word != blank)]
    df['n_chars'] = df.rec_word.str.len()
    df = df.reset_index()
    df['new_segment_start'] = (df['index'] - df['index'].shift(1,fill_value=df['index'].min())) > 1
    segment_ends = df[df.new_segment_start].index[1:].to_list() + [len(df)]
    return [Segment(file, start, end, df) for start, end in zip(df[df.new_segment_start].index, segment_ends)]


def read_df(path):
    if path.endswith('parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('feather'):
        df = pd.read_feather(path)
    else:
        raise ValueError(f'Type format is not supported {f}')
    return df


class Segment:
    def __init__(self, file, start, end, df):
        self.file = file
        self.start = start
        self.end = end

        self.n_words = end - start
        self.n_chars = df.iloc[start: end].n_chars.sum()
        self.duration = df.iloc[end-1].end - df.iloc[start].start
        self.avg_word_duration = self.duration / self.n_words
        self.avg_char_duration = self.duration / self.n_chars

    def get_alignment(self):
        df = read_df(self.file)
        return df.iloc[self.start: self.end]


# %%
if __name__ == '__main__':
    # %%
    warnings.simplefilter(action='ignore', category=FutureWarning)

    base_dir = '/lnet/express/work/people/stankov/alignment/new-parczech/baseline'
    selected_speakers_file_path = '/lnet/express/work/people/stankov/alignment/selecting_test_set/SELECTED-SPEAKERS-DEV-AND-TEST-people'
    speakers_info_file_path = '/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/parczech-3.0-speakers.tsv'
    align_data_dir = os.path.join(base_dir, 'align')
    output_data_dir = os.path.join(base_dir, 'punk_asr')
    # os.listdir(align_data_dir)
    blank = '<blank>'
    avg_char_dur_lb = 0.015
    avg_char_dur_ub = 0.25
    min_segment_len = 5
    min_segment_dur = 1.5
    max_segment_duration = 15
    min_offset_words = 1
    max_offset_words = 5
    binomial_p = 0.6
    binomial_n = 9

    # %%
    # extract selected speakers from the original ParCzech
    speakers_df = pd.read_csv(speakers_info_file_path, sep='\t').fillna('X')

    test_speakers = set()
    dev_speakers = set()
    with open(selected_speakers_file_path, 'r') as f:
        for line in f:
            set_type, speaker_name, *_ = line.split()
            speaker_name = speaker_name[2:-2]
            if 'dev' in set_type:
                dev_speakers.add(speaker_name)
            else:
                test_speakers.add(speaker_name)
    test_dev_speakers = test_speakers | dev_speakers
    # %%

    # raw_stats = dataset_statistics([os.path.join(align_data_dir, f) for f in sorted(os.listdir(align_data_dir))], zero_edit_dist=False)
    # zero_edit_dist_stats = dataset_statistics([os.path.join(align_data_dir, f) for f in sorted(os.listdir(align_data_dir))], zero_edit_dist=True)
    zero_edit_dist_segments = []

    for f in tqdm(sorted(os.listdir(align_data_dir))):
        zero_edit_dist_segments.extend(get_segments(os.path.join(align_data_dir, f)))

    # %%
    zero_edit_dist_segments_df = pd.DataFrame([
        dict(file=s.file, start=s.start, end=s.end, n_words=s.n_words, n_chars=s.n_chars, duration=s.duration, avg_word_dur=s.avg_word_duration, avg_char_duration=s.avg_char_duration)
        for s in zero_edit_dist_segments
    ])
    zero_edit_dist_segments_df.to_parquet('zero_edit_dist_segments.parquet')

    # %%
    # zero_edit_dist_segments_df.sort_values(by='n_chars').n_chars.plot()
    # plt.show()

    # %%

    np.random.seed(42)
    segment_durations = []
    useful_columns = ['segment_idx', 'rec_word', 'gold_word', 'speaker','start', 'end', 'dist']
    replace_punc = '|'.join([r'…', r'\\', r'’', r'@', r'¨', r'´', r'\|', r'\'', r'&'])
    replace_punc_with_comma = '|'.join([r'\)', r'\('])
    all_segments = []
    labels = []
    asr_dataset = []
    segments_stats = []

    for i, f in enumerate(tqdm(os.listdir(align_data_dir))):
        # if i == 500:
        #     break
        df = pd.read_feather(os.path.join(align_data_dir, f), columns=useful_columns)
        df['mp3'] = f.split('.')[0]

        # create labels
        df.gold_word = df.gold_word.str.replace(replace_punc, '')
        df.gold_word = df.gold_word.str.replace(replace_punc_with_comma, '')
        df['punc'] = '[blank]'
        df['upper'] = '[blank]'
        df['all_punc'] = df.gold_word.str.isalnum()
        df.loc[~df.gold_word.str[-1].str.isalnum(), 'punc'] = '[' + df.gold_word.str[-1] + ']'
        df.loc[df.gold_word.str[0].str.isupper(), 'upper'] = '[cap]'
        df.loc[df.gold_word.str[-1].str.isupper() & (df.gold_word.str[-1].str.len() > 1), 'upper'] = '[all_cap]'

        df['useful'] = False
        df.loc[(df.punc != '[blank]') | (df.upper != '[blank]'), 'useful'] = True

        # segments should have enough text and audio context
        raw_segments = get_punc_segments(df, min_segment_len, min_segment_dur)

        # process segments that belong to test/dev and have extra speakers
        segments = []
        for s in raw_segments:
            speakers = segment_get_speakers(s)
            if len(speakers & test_dev_speakers) == 0:
                segments.append(s)
                continue

            # if segment contains test or dev speaker check that it does not contain any other speakers
            if len(speakers & test_speakers) > 0:
                if len(speakers & test_speakers) == len(speakers):
                    segments.append(s)
                else:
                    segments.extend(split_by_speaker(s))

            if len(speakers & dev_speakers) > 0:
                if len(speakers & dev_speakers) == len(speakers):
                    segments.append(s)
                else:
                    segments.extend(split_by_speaker(s))

        # segemnts may be too long, cut segments if needed
        short_segments = []
        for s in segments:
            if len(s) < min_segment_len:
                continue

            if s[-1].end - s[0].start > max_segment_duration:
                short_segments.extend(split_long_segment(s, binomial_p, binomial_n))
            else:
                short_segments.append(s)

        # filter segments with all constraints
        # useful segments must have at least one word (except first one) with punctuation or have at least one capital letter at the beginning of the word
        useful_segments = [
            s for s in short_segments
            if min_segment_len <= len(s)
               and any(r.useful for r in s[1:])
               and (min_segment_dur <= (s[-1].end - s[0].start) <= max_segment_duration)
        ]

        if useful_segments != []:
            # 3 lines below can be used for dataset analysis
            segment_durations.extend([s[-1].end - s[0].start for s in useful_segments])
            labels.extend([r.punc for s in useful_segments for r in s])
            # all_segments.extend(useful_segments)

            df = pd.DataFrame([w for segment in useful_segments for w in segment])
            df = df[['Index', 'segment_idx', 'rec_word', 'gold_word', 'speaker', 'start', 'end', 'punc', 'upper']]
            df = df.rename(columns=dict(Index='word_idx', segment_idx='sentence_idx'))
            df['segment_idx'] = 0

            for j, segment in enumerate(useful_segments):
                indices = [w.Index for w in segment]
                df.loc[df.word_idx.isin(indices), 'segment_idx'] = j
                speakers = segment_get_speakers(segment)
                asr_example = dict(
                    idx=j,
                    mp3=f.split('.')[0],
                    start=segment[0].start,
                    end=segment[-1].end,
                    gold_text=' '.join(df.loc[df.word_idx.isin(indices), 'gold_word'].values.tolist()),
                    rec_text=' '.join(df.loc[df.word_idx.isin(indices), 'rec_word'].values.tolist()),
                    speakers=','.join(list(speakers)),
                    speakers_gender=','.join(speakers_df[speakers_df.id.isin(speakers)]['gender'].values.tolist()),
                    split='none',
                )
                if len(speakers & test_speakers) == len(speakers):
                    asr_example['split'] = 'test'
                elif len(speakers & dev_speakers) == len(speakers):
                    asr_example['split'] = 'dev'
                elif len(speakers & test_dev_speakers) == 0:
                    asr_example['split'] = 'train'

                asr_dataset.append(asr_example)

                n_words = len(indices)
                n_chars = len(asr_example['rec_text']) - n_words
                duration = asr_example['end'] - asr_example['start']

                segments_stats.append(dict(
                    split=asr_example['split'],
                    file=f,
                    n_words=n_words,
                    n_chars=n_chars,
                    duration=duration,
                    avg_word_dur=duration / n_words,
                    avg_char_duration=duration / n_chars

                ))
            # mp3_output_dir = os.path.join(output_data_dir, 'data', f.split('.')[0])
            # if not os.path.isdir(mp3_output_dir):
            #     os.makedirs(mp3_output_dir)

            # df.to_parquet(os.path.join(mp3_output_dir, 'alignment.parquet'))

    # pd.DataFrame(asr_dataset).to_parquet(os.path.join(output_data_dir, 'all_segments.parquet'))
    full_df = pd.DataFrame(asr_dataset)
    full_df['chars_per_sec'] = full_df.rec_text.str.len() / (full_df.end - full_df.start)
    subset_df = full_df[(full_df.chars_per_sec > 7) & (full_df.chars_per_sec < 25)]
    # subset_df.reset_index(drop=True).to_parquet(os.path.join(output_data_dir, 'clean_segments.parquet'))


    filtered_stats = pd.DataFrame(segments_stats)
    filtered_stats.to_parquet('filtered_segments_stats.parquet')
    # %%
    # %%