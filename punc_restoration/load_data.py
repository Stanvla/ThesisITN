# %%
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
from icecream import ic
from transformers import Wav2Vec2Model
import pytorch_lightning as pl
import os
import pandas as pd
from transformers import AutoTokenizer
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from pprint import pprint
import math
try:
    from args import get_args
except ModuleNotFoundError as e:
    from punc_restoration.args import get_args


BLANK = '[blank]'
punc_labels = [BLANK, '[,]', '["]', '[-]', '[.]', '[!]', '[;]', '[?]', '[:]']
punc_labels2int = {l: i for i, l in enumerate(punc_labels)}
int2punc_labels = {v: k for k, v in punc_labels2int.items()}

cap_labels = [BLANK, '[cap]', '[all_cap]']
cap_labels2int = {l: i for i, l in enumerate(cap_labels)}
int2cap_labels = {v: k for k, v in cap_labels2int.items()}

IGNORE_INDEX = -100
# https://huggingface.co/docs/transformers/tasks/token_classification


class FrameComputer:
    def __init__(self, model, params):
        # self.conv_layers = model.feature_extractor.conv_layers
        self.convs = []

        if params['audio_params']['backbone'] == 'w2v2' or params['audio_params']['backbone'] == 'baseline':
            for layer in model.feature_extractor.conv_layers:
                conv = layer.conv
                self.convs.append([
                    # conv.dilation[0] = 1, conv.kernel_size[0], conv.stride[0], conv.padding[0] = 0
                    conv.kernel_size[0], conv.stride[0]
                ])
        elif params['audio_params']['backbone'] == 'melspectrogram':
            spec_args = params['audio_params']['spec_args']
            self.convs.append([
                spec_args['n_fft'], spec_args['hop_length']
            ])
        else:
            raise ValueError(f'Unknown type of `audio_params.backbone` = {params["audio_params"]["backbone"]}')

        if 'cnn_model' in params['audio_params'] and params['audio_params']['backbone'] != 'baseline':
            audio_params = params['audio_params']['cnn_model']
            for k, s in zip(audio_params['kernels'], audio_params['strides']):
                self.convs.append([k, s])

    def __call__(self, wave_len):
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        for kernel_size, stride in self.convs:
            # conv = layer.conv
            # dilation, kernel_size, stride, padding = conv.dilation[0], conv.kernel_size[0], conv.stride[0], conv.padding[0]
            prev_len = wave_len

            if wave_len < kernel_size:
                return 0

            wave_len = math.floor(
                (wave_len - kernel_size) / stride + 1
            )
            # print(f'{wave_len} = [ ({prev_len} - {kernel_size}) / {stride} + 1]')
        # print()
        return wave_len


class TextData(Dataset):
    def __init__(self, args, split_type, tokenizer):
        self.args = args

        # main_df serves as index to get mp3_df/segment_df
        # main_df columns
        # ['idx', 'mp3', 'start', 'end', 'gold_text', 'rec_text', 'speakers','speakers_gender', 'split', 'chars_per_sec']
        self.main_df = pd.read_parquet(os.path.join(args['data_dir'], args['main_df']))
        if split_type is not None:
            self.main_df = self.main_df[self.main_df.split == split_type]
        self.main_df = self.main_df.reset_index(drop=True)

        self.data_path = os.path.join(args['data_dir'], 'data')
        self.tokenizer = tokenizer

    def get_mp3_folder(self, idx):
        mp3_name = self.main_df.iloc[idx].mp3
        return os.path.join(self.data_path, mp3_name)

    def tokenize_and_align_labels(self, segment_df):
        punc_targets_raw = [punc_labels2int[t] for t in segment_df.punc.values.tolist()]
        cap_targets_raw = [cap_labels2int[t] for t in segment_df.upper.values.tolist()]
        text_inputs_raw_lst = segment_df.rec_word.values.tolist()

        text_inputs_tokenized = self.tokenizer(text_inputs_raw_lst, is_split_into_words=True)
        word_ids = text_inputs_tokenized.word_ids(batch_index=0)
        # word ids without Nones
        padding_id = self.args['nn']['padding_id']
        positional_ids = [wi if wi is not None else padding_id for wi in word_ids]
        text_inputs_tokenized['positional_ids'] = positional_ids

        punc_targets_ids = []
        cap_targets_ids = []

        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                punc_targets_ids.append(IGNORE_INDEX)
                cap_targets_ids.append(IGNORE_INDEX)
            elif word_idx != prev_word_idx:
                # Only label the first token of a given word.
                punc_targets_ids.append(punc_targets_raw[word_idx])
                cap_targets_ids.append(cap_targets_raw[word_idx])
            else:
                punc_targets_ids.append(IGNORE_INDEX)
                cap_targets_ids.append(IGNORE_INDEX)
            prev_word_idx = word_idx

        text_inputs_tokenized['punc_targets'] = punc_targets_ids
        text_inputs_tokenized['cap_targets'] = cap_targets_ids
        return text_inputs_tokenized

    def __getitem__(self, idx):
        # get segment df using main_df
        segment_idx = self.main_df.iloc[idx].idx
        # segment_df columns
        # ['word_idx', 'sentence_idx', 'rec_word', 'gold_word', 'speaker', 'start', 'end', 'punc', 'upper', 'segment_idx'],
        mp3_df = pd.read_parquet(os.path.join(self.get_mp3_folder(idx), 'alignment.parquet'))
        segment_df = mp3_df[mp3_df.segment_idx == segment_idx]
        tokenized = self.tokenize_and_align_labels(segment_df)
        tokenized['idx'] = idx

        # starts and ends are needed for audio
        # however since we read the df we can add them
        tokenized['starts'] = (segment_df['start'] - segment_df['start'].min()).tolist()
        tokenized['ends'] = (segment_df['end'] - segment_df['start'].min()).tolist()
        return tokenized

    def get_verbose(self, idx):
        result = dict(
            norm_text=self.main_df.iloc[idx].rec_text,
            gold_text=self.main_df.iloc[idx].gold_text,
        )

        segment_idx = self.main_df.iloc[idx].idx
        mp3_df = pd.read_parquet(os.path.join(self.get_mp3_folder(idx), 'alignment.parquet'))
        segment_df = mp3_df[mp3_df.segment_idx == segment_idx]
        text_inputs_raw_lst = segment_df.rec_word.values.tolist()

        text_inputs_tokenized = self.tokenizer(text_inputs_raw_lst, is_split_into_words=True)
        word_ids = text_inputs_tokenized.word_ids(batch_index=0)

        result['word_ids'] = word_ids
        result['ends'] = (segment_df['end'] - segment_df['start'].min()).tolist()

        return result

    def __len__(self):
        return len(self.main_df)


# loads raw mp3 files that should be passed through Wav2Vec2
class RawAudioData(TextData):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.frame_computer = FrameComputer(
            model,
            kwargs['args'],
        )

        self.audio_shuffle = kwargs['args']['audio_params'].get('random_audio', False)
        self.sample_rate = 16_000

    def get_audio(self, idx):
        segment_idx = self.main_df.iloc[idx].idx
        mp3_folder = self.get_mp3_folder(idx)
        mp3_path = os.path.join(mp3_folder, f'{segment_idx}.mp3')
        mp3_tensor, sr = torchaudio.load(mp3_path)

        if mp3_tensor.shape[0] != 1:
            mp3_tensor = mp3_tensor.mean(0).unsqueeze(0)

        if self.audio_shuffle:
            new_indices = torch.randperm(mp3_tensor.shape[-1])
            mp3_tensor = mp3_tensor[:, new_indices]
        return mp3_tensor, mp3_path

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        mp3, mp3_path = self.get_audio(idx)
        result['raw_audio'] = mp3
        result['mp3_path'] = mp3_path
        # use starts and ends defined in Text Dataset

        result['features_starts'] = [self.frame_computer(s * self.sample_rate) for s in result['starts']]
        result['features_ends'] = [self.frame_computer(e * self.sample_rate) for e in result['ends']]

        starts = [self.frame_computer(s * self.sample_rate) for s in result['starts']]
        ends = [self.frame_computer(e * self.sample_rate) for e in result['ends']]
        audio_len = self.frame_computer(mp3.shape[-1])

        audio_word_positions = [-100] * audio_len
        audio_word_positions_cnt = [0] * audio_len
        for wi, (s, e) in enumerate(zip(starts, ends)):
            if s == e and e < audio_len:
                audio_word_positions_cnt[s] += 1
                audio_word_positions[s] = wi

            for idx in range(s, e):
                audio_word_positions_cnt[idx] += 1
                audio_word_positions[idx] = wi

        # now remove -100, which in general tells that there is a silence there
        # now silence will be the part of previous word

        audio_word_positions_clean = []
        prev_val = 0

        for v in audio_word_positions:
            if v != -100:
                audio_word_positions_clean.append(v)
            else:
                audio_word_positions_clean.append(prev_val)

            if v != -100:
                prev_val = v

        result['audio_positions'] = audio_word_positions_clean
        result['audio_positions_cnt'] = audio_word_positions_cnt
        result['raw_len'] = mp3.shape[-1]
        result['feature_len'] = result['features_ends'][-1]
        return result

    def get_verbose(self, idx):
        result = super().get_verbose(idx)
        result['feature_ends'] = [self.frame_computer(e * self.sample_rate) for e in result['ends']]
        return result


class AudioFeatureData(TextData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_audio_features(self, idx):
        segment_idx = self.main_df.iloc[idx].idx
        mp3_folder = self.get_mp3_folder(idx)
        audio_features_path = os.path.join(mp3_folder, f'{segment_idx}.parquet')
        audio_data = pd.read_parquet(audio_features_path).values
        return torch.from_numpy(audio_data)

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        result['audio_features'] = self.get_audio_features(idx)
        return result

class TextCollator:
    def __init__(self, pad_id, mask_pad_id=0, padding_emb=511):
        self.pad_id = pad_id
        self.mask_pad_id = mask_pad_id
        self.padding_emb = padding_emb

    def extract_field(self, batch, field):
        return [sample[field] for sample in batch]

    def pad_lst_tensor(self, lst, val, n):
        padding = [val] * n
        return torch.tensor(lst + padding)

    def __call__(self, batch):
        text_ids = self.extract_field(batch, 'input_ids')
        punc_targets = self.extract_field(batch, 'punc_targets')
        cap_targets = self.extract_field(batch, 'cap_targets')
        attn_masks = self.extract_field(batch, 'attention_mask')
        positional_ids = self.extract_field(batch, 'positional_ids')

        seq_lens = [len(x) for x in text_ids]
        max_seq_len = max(seq_lens)

        batch_text_ids, batch_punc_trgs, batch_cap_trgs, batch_msks, batch_positional = [], [], [], [], []
        for inp, punc, cap, msk, seq_len, pos in zip(text_ids, punc_targets, cap_targets, attn_masks, seq_lens, positional_ids):
            reminder = max_seq_len - seq_len
            batch_text_ids.append(self.pad_lst_tensor(inp, self.pad_id, reminder))
            batch_punc_trgs.append(self.pad_lst_tensor(punc, IGNORE_INDEX, reminder))
            batch_cap_trgs.append(self.pad_lst_tensor(cap, IGNORE_INDEX, reminder))
            batch_msks.append(self.pad_lst_tensor(msk, self.mask_pad_id, reminder))
            batch_positional.append(self.pad_lst_tensor(pos, self.padding_emb, reminder))

        return dict(
            text_ids=torch.stack(batch_text_ids),
            punc_trgs=torch.stack(batch_punc_trgs),
            cap_trgs=torch.stack(batch_cap_trgs),
            masks=torch.stack(batch_msks),
            seq_lens=torch.tensor(seq_lens),
            positional_ids_txt=torch.stack(batch_positional),
            indices=self.extract_field(batch, 'idx'),
            max_seq_len=max_seq_len,
        )


class RawAudioCollator(TextCollator):
    def __init__(self, **kwargs):
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16_000,
            padding_value=0,
            return_attention_mask=True,
        )
        super().__init__(**kwargs)

    def pad_multiple_lists(self, batch, list_name, padding_val=-100):
        lists = self.extract_field(batch, list_name)
        max_len = max([len(l) for l in lists])
        return torch.stack([
            self.pad_lst_tensor(l, padding_val, max_len - len(l)) for l in lists
        ])

    def __call__(self, batch):
        new_batch = super().__call__(batch)

        raw_audio_seq_lens = torch.tensor([x['raw_audio'].shape[-1] for x in batch])

        # https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481
        # https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad

        raw_audios = [{"input_values": x["raw_audio"].squeeze()} for x in batch]
        raw_audios_batch = self.feature_extractor.pad(
            raw_audios,
            padding=True,
            # pad_to_multiple_of=8,
            return_tensors="pt",
        )
        new_batch['raw_len'] = torch.tensor([x['raw_len'] for x in batch])
        new_batch['feature_len'] = torch.tensor([x['feature_len'] for x in batch])

        new_batch['audio'] = raw_audios_batch['input_values']
        new_batch['audio_mask'] = raw_audios_batch['attention_mask']

        new_batch['audio_sl'] = raw_audio_seq_lens
        new_batch['audio_path'] = [x['mp3_path'] for x in batch]

        new_batch['features_starts'] = self.pad_multiple_lists(batch, 'features_starts')
        new_batch['features_ends'] = self.pad_multiple_lists(batch, 'features_ends')

        new_batch['mp3_starts'] = self.pad_multiple_lists(batch, 'starts')
        new_batch['mp3_ends'] = self.pad_multiple_lists(batch, 'ends')
        new_batch['positional_ids_audio'] = self.pad_multiple_lists(batch, 'audio_positions', padding_val=self.padding_emb)
        new_batch['positional_ids_audio_cnt'] = self.pad_multiple_lists(batch, 'audio_positions_cnt', padding_val=self.padding_emb)

        return new_batch


class AudioFeatureCollator(TextCollator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, batch):
        new_batch = super().__call__(batch)

        seq_lens = [x['audio_features'].shape[0] for x in batch]
        M = max(seq_lens)

        new_batch['audio_features'] = torch.stack([
            F.pad(x['audio_features'], (0, 0, M - sl, 0)) for x, sl in zip(batch, seq_lens)
        ])

        new_batch['audio_sl'] = seq_lens
        return new_batch


class DataPL(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args['robeczech_path'], add_prefix_space=True)
        self.train_dataset = TextData(args, 'train', self.tokenizer)
        self.test_dataset = TextData(args, 'test', self.tokenizer)
        self.dev_dataset = TextData(args, 'dev', self.tokenizer)

    def _get_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=TextCollator(pad_id=self.tokenizer.pad_token_id, padding_emb=self.args['nn']['padding_id']),
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args['dataset_workers']
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, self.args['batch_size'])

    def val_dataloader(self):
        return self._get_dataloader(self.dev_dataset, self.args['batch_size'])

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, self.args['batch_size'])


class AudioDataPL(DataPL):
    def __init__(self, args, w2v_model):
        super().__init__(args)
        # w2v_model = Wav2Vec2Model.from_pretrained(args['audio_params']['model_weights_path'])
        self.train_dataset = RawAudioData(w2v_model, args=args, split_type='train', tokenizer=self.tokenizer)
        self.test_dataset = RawAudioData(w2v_model, args=args, split_type='test', tokenizer=self.tokenizer)
        self.dev_dataset = RawAudioData(w2v_model, args=args, split_type='dev', tokenizer=self.tokenizer)

    def _get_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=RawAudioCollator(pad_id=self.tokenizer.pad_token_id, padding_emb=self.args['nn']['padding_id']),
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args['dataset_workers']
        )


class AudioFeatureDataPL(DataPL):
    def __init__(self, args):
        super().__init__(args)
        self.train_dataset = AudioFeatureData(args=args, split_type='train', tokenizer=self.tokenizer)
        self.test_dataset = AudioFeatureData(args=args, split_type='test', tokenizer=self.tokenizer)
        self.dev_dataset = AudioFeatureData(args=args, split_type='dev', tokenizer=self.tokenizer)

    def _get_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=AudioFeatureCollator(pad_id=self.tokenizer.pad_token_id),
            pin_memory=torch.cuda.is_available(),
            num_workers=self.args['dataset_workers']
        )


# %%
if __name__ == '__main__':
    # %%
    torch.set_printoptions(precision=3, linewidth=260)

    params = get_args()
    audio_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])

    audio_data_pl = AudioDataPL(params, audio_model)
    audio_dev_dataset = audio_data_pl.dev_dataset


    for i in range(len(audio_dev_dataset)):
        ex = audio_dev_dataset[i]
        print(ex.keys())
        if i == 2:
            break


    # %%
    # audio_data_pl = AudioDataPL(params)
    # dev_loader = audio_data_pl.val_dataloader()
    # train_loader = audio_data_pl.train_dataloader()
    # for i, batch in enumerate(tqdm(train_loader)):
    #     if batch['audio'].shape[-1] % 8 != 0:
    #         print(i)
    #
    # # %%
    # data_pl = DataPL(params)
    # train_loader = data_pl.train_dataloader()
    # test_loader = data_pl.test_dataloader()
    # dev_loader = data_pl.val_dataloader()
    # # %%
    # from tqdm import tqdm
    # for x in tqdm(train_loader):
    #     pass
    # # %%
