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

    np.random.seed(42)
    segment_durations = []
    useful_columns = ['segment_idx', 'rec_word', 'gold_word', 'speaker','start', 'end', 'dist']
    replace_punc = '|'.join([r'…', r'\\', r'’', r'@', r'¨', r'´', r'\|', r'\'', r'&'])
    replace_punc_with_comma = '|'.join([r'\)', r'\('])
    all_segments = []
    labels = []
    asr_dataset = []

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

            mp3_output_dir = os.path.join(output_data_dir, 'data', f.split('.')[0])
            if not os.path.isdir(mp3_output_dir):
                os.makedirs(mp3_output_dir)

            df.to_parquet(os.path.join(mp3_output_dir, 'alignment.parquet'))

    pd.DataFrame(asr_dataset).to_parquet(os.path.join(output_data_dir, 'all_segments.parquet'))
    full_df = pd.DataFrame(asr_dataset)
    full_df['chars_per_sec'] = full_df.rec_text.str.len() / (full_df.end - full_df.start)
    subset_df = full_df[(full_df.chars_per_sec > 7) & (full_df.chars_per_sec < 25)]
    subset_df.reset_index(drop=True).to_parquet(os.path.join(output_data_dir, 'clean_segments.parquet'))

    # %%

    # %%

    bins = np.linspace(
        min(segment_durations),
        max(segment_durations),
        50,
    )
    plt.xlim([min(segment_durations) - 5, max(segment_durations) + 5])
    plt.hist(segment_durations, bins=bins)
    plt.title(f'binomial {binomial_p} {binomial_n}, {sum(segment_durations) / 3600:.4f} {len(segment_durations)}')
    plt.yscale('log')
    plt.show()


    # %%
    print(set(subset_df[subset_df.speakers.isin(test_speakers)].speakers.unique()) == set(full_df[full_df.speakers.isin(test_speakers)].speakers.unique()))
    print(set(subset_df[subset_df.speakers.isin(dev_speakers)].speakers.unique()) == set(full_df[full_df.speakers.isin(dev_speakers)].speakers.unique()))

    # %%
    plt.plot(range(len(full_df)), sorted(full_df.chars_per_sec.values.tolist()))
    plt.show()

    # %%
    plt.plot(range(len(subset_df)), sorted(subset_df.chars_per_sec.values.tolist()))
    plt.show()
    # %%
    # %%
    print(sum(full_df.duration) / 3600)
    # %%

    # show most common labels
    label_counter = Counter(labels)
    label_counter.most_common()
    # %%
    display_segment(all_segments[4])
    # %%
    # view test/dev durations
    speakers_durations = {}
    for s in all_segments:
        speakers = segment_get_speakers(s)
        duration = s[-1].end - s[0].start
        for speaker in speakers:
            if speaker not in speakers_durations:
                speakers_durations[speaker] = duration
            else:
                speakers_durations[speaker] += duration

    test_duration = sum(speakers_durations[speaker] for speaker in test_speakers)
    dev_duration = sum(speakers_durations[speaker] for speaker in dev_speakers)
    print(test_duration / 3600, dev_duration / 3600)

    # %%
    multiple_speakers = {}
    for i, s in enumerate(all_segments):
        speakers = segment_get_speakers(s)
        if len(speakers) > 1:
            if len(speakers) not in multiple_speakers:
                multiple_speakers[len(speakers)] = [s]
            else:
                multiple_speakers[len(speakers)].append(s)

    for k, v in multiple_speakers.items():
        print(k, len(v))

    # %%
    for k, segments_lst in multiple_speakers.items():
        print(k)
        for s in segments_lst:

            speakers = segment_get_speakers(s)
            if len(speakers & test_speakers) > 0:
                print('test', speakers, speakers & test_speakers)
            if len(speakers & dev_speakers) > 0:
                print('dev', speakers, speakers & dev_speakers)
            if len(speakers & dev_speakers & test_speakers) > 0:
                print('test-dev', speakers)

    # %%
    display_segment(multiple_speakers[2][5])
    # %%
    short_segments = [s for s in all_segments if s[-1].end - s[0].start < 15]
    sum(s[-1].end - s[0].start for s in short_segments) / sum(s[-1].end - s[0].start for s in all_segments)

    # %%
    long_segments = [s for s in all_segments if s[-1].end - s[0].start > duration_ub]

    # len(long_segments) / len(all_segments)
    # sum(s[-1].end - s[0].start for s in long_segments) / sum(s[-1].end - s[0].start for s in all_segments)
    divided_long_segments = []
    for i, s in enumerate(long_segments):
        if i == 10:
            break
        short_segments = []
        short_segment = []
        offset = np.random.randint(3, 20)
        useful_flag = False
        print(offset)
        for w in s:
            if useful_flag:
                offset -= 1

            useful_flag = useful_flag or w.useful
            short_segment.append(w)

            if offset == 0:
                short_segments.append(short_segment)
                display_segment(short_segment)
                short_segment = []
                offset = np.random.randint(3, 20)
                useful_flag = False
                print(offset)

        # last short segment may be missed
        if short_segment != []:
            # last segment may not contain punc
            if any(w.useful for w in short_segment):
                short_segments.append(short_segment)
            else:
                short_segments[-1].extend(short_segment)

            display_segment(short_segments[-1])

        print('====='*22)
    # %%
    # %%

    # segments = get_punc_segments(word_alignments[-1])
    durations = []
    segment_max_duration = 2
    segment_min_duration = 0
    cnt = 0
    for s in all_segments:

        start = s[0].start
        end = s[-1].end
        if end - start > segment_max_duration or end - start < segment_min_duration:
            continue
        durations.append(end - start)
        cnt += 1
        if cnt >= 29:
            continue
        display_segment(s)
    print(sum(durations) / 3600)
    # %%
