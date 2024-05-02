import os

import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from moviepy.editor import *
from Bio import pairwise2
from Levenshtein import distance
try:
    from wav2vec2_ctc.extract_time_logits import RecognizedWord, RecognizedSegment, force_alignment
except:
    from extract_time_logits import RecognizedWord, RecognizedSegment, force_alignment
from tqdm import tqdm
import argparse
import logging


# choose best segment len, so the last input is not too short
def set_segment_len(start_range, end_range, mp3_frames_cnt):
    best_segment_len = start_range
    best_segment_len_reminder = 0
    if mp3_frames_cnt / params.sr < start_range:
        return start_range

    for segment_len in range(start_range, end_range):
        total_offset = 0
        segments = []

        while True:
            if total_offset >= mp3_frames_cnt:
                break
            segments.append(min(mp3_frames_cnt, total_offset + segment_len * params.sr) - total_offset)
            total_offset = min(mp3_frames_cnt, total_offset + (segment_len - params.overlap_len_sec) * params.sr)

        segments = [s / params.sr for s in segments]

        if min(segments) > best_segment_len_reminder:
            best_segment_len = segment_len
            best_segment_len_reminder = min(segments)

    return best_segment_len


def get_word_times(word_segments, trellis, waveform, sample_rate, offset):
    # waveform = torch.tensor(waveform).unsqueeze(0)
    waveform = waveform.clone().unsqueeze(0)
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    words = []
    for i in range(len(word_segments)):
        word = word_segments[i]
        x0 = int(ratio * word.start) / sample_rate + offset
        x1 = int(ratio * word.end) / sample_rate + offset
        words.append(RecognizedWord(word.label, x0, x1))
    return words


def initialize():
    model = Wav2Vec2ForCTC.from_pretrained(params.model_name)
    processor = Wav2Vec2Processor.from_pretrained(params.processor_name)
    resampler = torchaudio.transforms.Resample(params.sr, params.target_sr)

    vocabulary = processor.tokenizer.get_vocab()
    exclude_voc = ['<s>', '</s>']
    vocabulary = {k: v for k, v in vocabulary.items() if k not in exclude_voc}
    vocabulary_lst = [''] * len(vocabulary)
    for k, v in vocabulary.items():
        vocabulary_lst[v] = k

    decoder = build_ctcdecoder(labels=vocabulary_lst, kenlm_model_path=params.lm_path, alpha=params.alpha, beta=params.beta)
    return model, processor, resampler, vocabulary, decoder


def get_model_inputs(source_tensor, resampler):
    total_offset = 0
    samples = []

    while True:
        if total_offset >= source_tensor.shape[1]:
            break
        x = source_tensor[0, total_offset: min(source_tensor.shape[1], total_offset + params.segment_len_sec * params.sr)]
        total_offset = total_offset + min((params.segment_len_sec - params.overlap_len_sec) * params.sr, source_tensor.shape[1] - total_offset)
        samples.append(x)

    return [processor(resampler(x), sampling_rate=params.target_sr, return_tensors='pt', padding=True) for x in samples], samples


class Params:
    sr: int = 48000
    target_sr: int = 16000
    beam_width: int = 150
    segment_len_sec: int = 27
    overlap_len_sec: int = 5
    discard_words: int = 2
    alpha: float = 0.75
    beta: float = 1.0
    lm_path: str = '/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/kenlm_model.arpa'

    def __init__(self, model_name_str, mp3_source_path, processor_name_str=None):
        self.model_name = model_name_str
        self.processor_name = model_name_str if processor_name_str is None else processor_name_str
        self.mp3_source = mp3_source_path


class GoldWord:
    not_puct = ['§', '%', '=', '/', '+']

    def __init__(self, line):
        args = line[: -1].split('\t')
        self.word = args[0]
        self.id = args[1]
        self.speaker = args[2]

    def __int__(self, word, id, speaker):
        self.word = word
        self.id = id
        self.speaker = speaker

    def ignore_word(self, voc):
        missing = [ch for ch in self.word if ch not in voc and not ch.isdigit()]
        return missing != []

    @property
    def norm_word(self):
        return ''.join([ch for ch in self.word.lower() if ch.isalnum()])

    @property
    def word_id(self):
        return int(self.id.split('.')[-1].replace('w', ''))

    def is_punc(self):
        return all(not ch.isalnum() and ch not in self.not_puct for ch in self.word)

    def __repr__(self):
        return f'word :: {self.word:15} id :: {self.id}'


class AlignedSegment:
    blank = '<blank>'

    def __init__(self, words_lst):
        self.gold_words = [g for g, _ in words_lst]
        self.rec_words = [r for _, r in words_lst]
        self.empty = all(w is None for w in self.gold_words)

    def display_alignment(self):
        for rec_word, gold_word in zip(self.rec_words, self.gold_words):
            r_str = self.blank if rec_word is None else rec_word.word
            g_str = self.blank if gold_word is None else gold_word.word
            g_str_norm = self.blank if gold_word is None else gold_word.norm_word
            if r_str == g_str_norm:
                print(f'{g_str:^30}')
            else:
                print(f'{r_str:20} {g_str:20}')

    @property
    def segment_start(self):
        for w in self.rec_words:
            if w is not None:
                return w.start
        return -1

    @property
    def segment_end(self):
        for w in self.rec_words[::-1]:
            if w is not None:
                return w.end
        return -1

    @property
    def duration(self):
        if self.segment_start == -1 or self.segment_end == -1:
            return -1
        else:
            return self.segment_end - self.segment_start

    @property
    def edit_distance(self):
        return distance(self.rec_text, self.gold_text(norm=True))

    def contain_numbers(self):
        return any(ch.isdigit() for ch in self.gold_text(norm=False))

    def gold_text(self, norm):
        if norm:
            return ' '.join([w.norm_word for w in self.gold_words if w is not None])
        else:
            return ' '.join([w.word for w in self.gold_words if w is not None])

    @property
    def rec_text(self):
        return ' '.join([w.word for w in self.rec_words if w is not None])

    def edit_distance_norm(self, normalization):
        # normalization can be [max, recognized, gold]
        if normalization == 'max':
            return self.edit_distance / max(len(self.rec_text), len(self.gold_text(norm=True)))
        elif normalization == 'recognized':
            return self.edit_distance / len(self.rec_text)
        elif normalization == 'gold':
            return self.edit_distance / len(self.gold_text(norm=True))
        else:
            raise ValueError(f'Normalization can be [max, recognized, gold], but `{normalization}` was passed.')

    def dump_to_dicts(self, segment_idx):
        alignment_with_info = []
        for rec_word, gold_word in zip(self.rec_words, self.gold_words):
            alignment_dict = dict(
                segment_idx=segment_idx,
                rec_word=self.blank if rec_word is None else rec_word.word,
                gold_word=self.blank if gold_word is None else gold_word.word,
                gold_word_norm=self.blank if gold_word is None else gold_word.norm_word,
                gold_word_is_alpha=False if gold_word is None else gold_word.norm_word.isalpha(),
                gold_word_contains_digit=False if gold_word is None else any(ch.isdigit() for ch in gold_word.norm_word),
                id=self.blank if gold_word is None else gold_word.id,
                speaker=self.blank if gold_word is None else gold_word.speaker,
                start=-1 if rec_word is None else rec_word.start,
                end=-1 if rec_word is None else rec_word.end,
            )

            if rec_word is not None and gold_word is not None:
                alignment_dict['dist'] = distance(rec_word.word, gold_word.norm_word)
            elif rec_word is not None:
                alignment_dict['dist'] = len(rec_word.word)
            elif gold_word is not None:
                alignment_dict['dist'] = len(gold_word.norm_word)
            else:
                if do_logging:
                    logging.error('Blank distance, empty string aligned to empty string')
                alignment_dict['dist'] = 10000
            alignment_dict['dist'] = f'{alignment_dict["dist"]}'

            alignment_with_info.append(alignment_dict)
        return alignment_with_info

    def contains_non_alphanum(self):
        return any(not w.norm_word.isalnum() for w in self.gold_words if w is not None)

    def summary(self, segment_idx, mp3_name):
        speakers = sorted(list(set([w.speaker for w in self.gold_words if w is not None])))
        return dict(
            mp3_name=mp3_name,
            segment_idx=segment_idx,
            global_start=self.segment_start,
            global_end=self.segment_end,
            duration=self.segment_end - self.segment_start,
            edit_distance=self.edit_distance,
            start_end_align=self.rec_words[0] is not None and self.rec_words[-1] is not None,
            cnt_speakers=len(speakers),
            speakers=speakers,
            norm_contains_non_alphanum=self.contains_non_alphanum(),
            contains_digits=self.contain_numbers(),
            gold_text=self.gold_text(norm=False),
            gold_text_norm=self.gold_text(norm=True),
            recognized_text=self.rec_text,
        )


def reduce_vertical(gold_vert_lst):
    # by default punctuation is a separate word, append punctuation to the previous word
    new_gold_vert_lst = []
    for gold_word in gold_vert_lst:
        if gold_word.is_punc() and new_gold_vert_lst != []:
            new_gold_vert_lst[-1].word += gold_word.word
        else:
            new_gold_vert_lst.append(gold_word)
    return new_gold_vert_lst


def get_logits(model_inputs, n=None):
    logging.info('get logits')
    if n is None:
        n = len(model_inputs)
    n = min(n, len(model_inputs))
    logits_lst = []
    for i in range(n):
        with torch.no_grad():
            logits = model(**model_inputs[i]).logits.cpu().detach()
            logits_lst.append(logits)
    return logits_lst


def align_gold_rec(logits_lst, mp3_example_name):
    logging.info('align gold rec')
    # ..... read gold vertical ....
    gold_vertical_path = os.path.join("/lnet/express/work/people/stankov/alignment/results/full/merged", f'{mp3_example_name}.vert')
    with open(gold_vertical_path, 'r') as f:
        gold_vertical_lst = [GoldWord(line) for line in f]
    gold_vertical_lst = reduce_vertical(gold_vertical_lst)

    # ..... get words and their times from logits ....
    logging.info('align gold rec :: decoding')
    word_times = []
    processed_time_pure = 0
    for i, logits in enumerate(logits_lst):
        processed_time_pure += inputs[i]['input_values'].shape[1]
        if i > 0:
            processed_time_pure -= params.overlap_len_sec * params.target_sr

        decoded = decoder.decode(logits.numpy()[0], beam_width=params.beam_width)
        trellis, segments = force_alignment(logits, decoded, vocabulary, processor)
        words_with_time_lst = get_word_times(segments, trellis, orig_samples[i], params.sr, (params.segment_len_sec - params.overlap_len_sec) * i)
        word_times.append(words_with_time_lst)

    if processed_time_pure / params.target_sr != mp3_tensor.shape[1] / params.sr:
        if do_logging:
            logging.error(f'Recognized duration :: {processed_time_pure / params.target_sr} does not match mp3 duration {mp3_tensor.shape[1] / params.sr}')

    # ..... since audio was recognized with overlaps remove overlapping words ....
    logging.info('align gold rec :: merging pieces into single recognized vertical')
    recognized_vertical = []
    end_time = -1
    segment_start = 0
    for i, segment in enumerate(word_times):
        tmp_end = 0
        for i, w in enumerate(segment):
            if w.start > end_time:
                # `(len(segment) - i > params.discard_words)` leave only words that are not "at the end" == discard last `params.discard_words`
                # or
                # `(w.end - segment_start) <= params.segment_len_sec - params.overlap_len_sec / 2` leave words with timestamp smaller than
                # segment length - half of the overlap
                if (len(segment) - i > params.discard_words) or ((w.end - segment_start) <= params.segment_len_sec - params.overlap_len_sec / 2):
                    tmp_end = w.end
                    recognized_vertical.append(w)
        segment_start += params.segment_len_sec - params.overlap_len_sec
        end_time = tmp_end

    # ..... create alignment between recognized and gold verticals ....
    logging.info('align gold rec :: creating alignment ')

    gap_char = '-'
    alignment = pairwise2.align.globalcx(
        [w.word for w in recognized_vertical],
        [w.norm_word for w in gold_vertical_lst],
        lambda x, y: max(len(x), len(y)) - distance(x, y),
        gap_char=[gap_char],
    )[0]

    # ..... split alignment into segments according to the gold word id ....
    logging.info('align gold rec :: spliting alignnment into segments')

    # the reason to create a temporary segment list is the following
    # imagine one segment ends and last recognized word is aligned to the gold word,
    # but next recognized word is not aligned to blank symbol
    # in this case this word will stay with already finished segment
    # and this is undesired
    aligned_segments, segment_lst, tmp_segment_lst = [], [], []
    gold_idx, rec_idx, prev_word_id, undefined = 0, 0, -1, True

    for rec_word_str, gold_word_str in zip(alignment.seqA, alignment.seqB):
        gold_word = gold_vertical_lst[gold_idx] if gold_idx >= 0 and gold_word_str != gap_char else None
        rec_word = recognized_vertical[rec_idx] if rec_idx >= 0 and rec_word_str != gap_char else None

        if gold_word is not None and undefined:
            undefined = False

        if gold_word is not None and gold_word.word_id < prev_word_id:
            aligned_segment = AlignedSegment(segment_lst)
            if not aligned_segment.empty:
                aligned_segments.append(aligned_segment)
            segment_lst = []

        # if prev_word_id is undefined
        if not undefined:
            tmp_segment_lst.append([gold_word, rec_word])

        if rec_word_str != gap_char:
            rec_idx += 1

        if gold_word_str != gap_char:
            gold_idx += 1
            prev_word_id = gold_word.word_id
            segment_lst += [[x, y] for x, y in tmp_segment_lst]
            tmp_segment_lst = []

    return recognized_vertical, aligned_segments, alignment


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/models/15_epochs/model", type=str)
parser.add_argument("--processor_path", default="/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/models/15_epochs/preprocessor", type=str)
parser.add_argument("--mp3_source", default="/lnet/work/people/kopp/ParCzech/psp.cz/mp3/original_files/www.psp.cz/eknih/2013ps/audio/2013/11/25/2013112513581412.mp3", type=str)
parser.add_argument("--output_dir", default="/lnet/express/work/people/stankov/alignment/new-parczech/baseline", type=str)
parser.add_argument("--job_idx", type=int)


# %%
if __name__ == '__main__':
    # %%
    args = parser.parse_args([] if "__file__" not in globals() else None)

    do_logging = False
    mp3_name = os.path.basename(args.mp3_source).replace('.mp3', '')

    log_dir = os.path.join(args.output_dir, 'python_logs')
    asr_dir = os.path.join(args.output_dir, 'asr')
    align_dir = os.path.join(args.output_dir, 'align')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(asr_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)

    if log_dir != "":
        do_logging = True
        log_file = os.path.join(log_dir, f'{mp3_name}.log')
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', datefmt='%H:%M:%S %d.%m.%Y')
        logging.info(f'job {args.job_idx}')
        logging.info(args.mp3_source)
        logging.info(mp3_name)
        logging.info('start')

    params = Params(
        model_name_str="/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/models/15_epochs/model",
        mp3_source_path=args.mp3_source,
        processor_name_str="/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/models/15_epochs/preprocessor"
    )

    mp3_tensor, sr = torchaudio.load(params.mp3_source)
    params.sr = sr

    model, processor, resampler, vocabulary, decoder = initialize()

    params.segment_len_sec = set_segment_len(27, 33, mp3_tensor.shape[1])
    inputs, orig_samples = get_model_inputs(mp3_tensor, resampler)
    logits = get_logits(inputs)

    rec_vert, align_segments, alignm = align_gold_rec(logits, mp3_name)
    # save alignment info for punctuation restoration: start, end, recognized word, gold word, normalized word, speaker, ...
    word_level_df = pd.DataFrame([d for i, s in enumerate(align_segments) for d in s.dump_to_dicts(i)])
    word_level_df.to_feather(os.path.join(align_dir, f'{mp3_name}.feather'))

    #  save segment info for ASR training:
    #       `segment.summary()`
    segment_level_df = pd.DataFrame([s.summary(i, mp3_name) for i, s in enumerate(align_segments)])
    segment_level_df.to_feather(os.path.join(asr_dir, f'{mp3_name}.feather'))

    if do_logging:
        logging.info('done')
    # %%
    for i, s in enumerate(align_segments):
        print(i, s.duration)
    # %%
    # for i, (rec, gold) in enumerate(zip(alignm.seqA, alignm.seqB)):
    #     if i == 199: break
    #     print(f'{rec:20} {gold:20}')
    # # %%
    # align_segments[0].display_alignment()
    # # %%
    # print(align_segments[0].dump_to_string())
    # # %%
    # print(align_segments[0].dump_to_json())
    # # %%
    # # with open(args.recognized_file, 'w') as f:
    # #     f.write('word\tstart\t\end')
    # #     for w in recognized_vertical:
    # #         f.write(f'{w.word}\t{w.start:.4f}\t{w.end:.4f}\n')
    #
    #
    # # print(good_duration)
    # # print(good_duration / (mp3_tensor.shape[1] / sr))
    # # print(good_duration / (mp3_tensor.shape[1] / sr) * 3000)
    # # %%
    # good_threshold = 0.0
    # bad_threshold = 0.7
    # #
    # good_segments = [s for s in align_segments if s.edit_distance_norm(normalization='max') <= good_threshold]
    # bad_segments = [s for s in align_segments if s.edit_distance_norm(normalization='max') >= bad_threshold]
    # #
    # len(good_segments) / len(align_segments)
    # # %%
    # for s in align_segments:
    #     print(f'start {s.segment_start:.4f}')
    #     s.display_alignment()
    #     print(f'edit dist max {s.edit_distance_norm(normalization="max"):.4f}')
    #     print(f'end {s.segment_end:.4f}, duration {s.duration:.4f}')
    #     print('---'*30)
    #
    # # %%
    # for s in align_segments:
    #     if s.contain_numbers():
    #         print(f'start {s.segment_start:.4f}')
    #         s.display_alignment()
    #         print(f'edit dist max {s.edit_distance_norm(normalization="max"):.4f}')
    #         print(f'end {s.segment_end:.4f}, duration {s.duration:.4f}')
    #         print('---'*30)
    #
    # # for rec_word, gold_word in zip(alignment.seqA, alignment.seqB):
    # #     if rec_word == gold_word:
    # #         print(f'{rec_word:^30}')
    # #     else:
    # #         print(f'{rec_word:20} {gold_word:20}')
    #
    # if do_logging:
    #     logging.info('done')
    # %%
