from typing import Union, Tuple

import pandas as pd
import transformers.models.wav2vec2
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from transformers import AutoModel, Wav2Vec2Model
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
import logging
import os
import argparse
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pprint import pprint
from tqdm import tqdm
try:
    from args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from cls_head import ClassificationHead
    from load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL
    from multi_modal_att import MMEncoder
    from metrics import MetricsComp
    from audio_model import AudioEncoder
    from sequence_labeling import TextBasedModel, FullModel
except ModuleNotFoundError as e:
    from punc_restoration.metrics import MetricsComp
    from punc_restoration.audio_model import AudioEncoder
    from punc_restoration.multi_modal_att import MMEncoder
    from punc_restoration.args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from punc_restoration.cls_head import ClassificationHead
    from punc_restoration.load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL
    from punc_restoration.sequence_labeling import TextBasedModel, FullModel

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='sound')
parser.add_argument("--bs", type=int, default=200)
parser.add_argument('--model_name', type=str)
parser.add_argument("--time_ckp", type=str)
parser.add_argument("--time_cfg", type=str)
parser.add_argument("--model_type", type=str, default='last')
parser.add_argument("--debug", type=bool, default=False)


def attention_matrices(trained_model, data_loader, dataset):
    def plot_attn(f, ax, example, title_font_size, basic_font_size):
        title = 'Text length {text_len}, Audio duration {audio_len:3.2f} sec., Loss {loss:1.4f}'.format(text_len=example['txt_len'], audio_len=example['raw_len'], loss=example['loss'])
        cax = ax.matshow(example['cr_attn'], cmap='bone', aspect='auto')
        # Set up axes
        ax.set_xticks(range(example['snd_len']))
        ax.set_xticklabels(labels=example['snd_ticks'])
        ax.tick_params(axis='x', which='major', rotation=45)

        ax.set_yticks(range(len(example['txt_ticks'])))
        ax.set_yticklabels(labels=example['txt_ticks'])
        ax.set_title(title)

        # changing font size
        for item in ([ax.title]):
            item.set_fontsize(title_font_size)

        for item in ([ax.xaxis.label] + ax.get_xticklabels()):
            item.set_fontsize(basic_font_size - 2)

        for item in ([ax.yaxis.label] + ax.get_yticklabels()):
            item.set_fontsize(basic_font_size)

        f.colorbar(cax, ax=ax)

    def worst_loss(lst, start=0, end=-1):
        wl = 0
        wl_ex = None
        for x in lst[start: end]:
            if x['loss'] > wl:
                wl = x['loss']
                wl_ex = x
        return wl_ex

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = trained_model.to(device)

    attentions = []

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if cmd_args.debug and i == 10:
            break

        batch_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        punc_trgs, case_trgs, seq_lens, indices, raw_lens = batch['punc_trgs'], batch['cap_trgs'], batch['seq_lens'], batch['indices'], batch['raw_len']

        with torch.no_grad():
            outputs_punc, outputs_case, text_attn, cross_attn = trained_model(batch_gpu)
            punc_loss, punc_preds, _, punc_losses = trained_model.get_loss_metrics(outputs_punc.to(device), punc_trgs.to(device))
            case_loss, case_preds, _, case_losses = trained_model.get_loss_metrics(outputs_case.to(device), case_trgs.to(device))

        text_attn = torch.mean(text_attn.detach(), dim=1)
        for punc_l, case_l, idx, cr_attn, t_attn, raw_len \
                in zip(
                    punc_losses,
                    case_losses,
                    indices,
                    cross_attn.cpu().numpy(),
                    text_attn.cpu().numpy(),
                    raw_lens.cpu().numpy(),
        ):
            loss = 0.5 * punc_l + 0.5 * case_l
            data = dataset.get_verbose(idx)

            # audio ticks, audio len
            ends = data['feature_ends']

            audio_len = ends[-1] + 1
            audio_ticks = [list() for _ in range(audio_len)]
            for end, word in zip(ends, data['gold_text'].split()):
                audio_ticks[end].append(word)

            new_audio_ticks = []
            max_tick_len = 20
            for tlst in audio_ticks:
                new_tick = ""
                new_tick_len = 0
                for t in tlst:
                    if new_tick_len + len(t) + 1 < max_tick_len:
                        new_tick += f' {t}'
                        new_tick_len += 1 + len(t)
                    else:
                        new_tick_len = 0
                        new_tick += f'\n{t}'
                new_audio_ticks.append(new_tick)

            # text ticks, text len
            text_ticks, prev_word_id, gold_text_idx = [], None, 0
            for word_id in data['word_ids']:
                if word_id is None:
                    text_ticks.append('')
                    continue
                if prev_word_id != word_id:
                    text_ticks.append(data['gold_text'].split()[gold_text_idx])
                    gold_text_idx += 1
                else:
                    text_ticks.append('')
                prev_word_id = word_id
            text_len = len(text_ticks)

            cr_attn = cr_attn[:text_len, :audio_len]
            t_attn = t_attn[:text_len, :text_len]
            attentions.append(dict(
                loss=loss.cpu().numpy(),
                txt_len=len([t for t in text_ticks if t != '']),
                snd_len=audio_len,
                snd_ticks=new_audio_ticks,
                txt_ticks=text_ticks,
                cr_attn=cr_attn,
                t_attn=t_attn,
                raw_len=raw_len / 16_000,
            ))

    len_sorted_attns = list(sorted(attentions, key=lambda d: d['txt_len']))
    q1, q2, q3 = math.floor(len(attentions) * 0.25), math.floor(len(attentions) * 0.5), math.floor(len(attentions) * 0.75)

    q1_bad = worst_loss(len_sorted_attns, start=0, end=q1)
    q2_bad = worst_loss(len_sorted_attns, start=q1, end=q2)
    q3_bad = worst_loss(len_sorted_attns, start=q2, end=q3)

    q4_bad = worst_loss(len_sorted_attns, start=q3, end=-1)
    qs = [[q1_bad, q2_bad], [q3_bad, q4_bad]]
    # https://stackoverflow.com/a/73285228


    # import matplotlib.gridspec as gridspec
    # fig = plt.figure(figsize=(28, 12))
    # gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 5])
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])
    # axes = [ax1, ax2, ax3, ax4]
    # for a, x in zip(axes, [q1_bad, q2_bad, q3_bad, q4_bad]):
    #     plot_attn(fig, a, x, 17, 15)

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), gridspec_kw={'height_ratios': [1, 1.5, 2, 3]})
    for a, x in zip(axes, [q1_bad, q2_bad, q3_bad, q4_bad]):
        plot_attn(fig, a, x, 14, 11)

    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         plot_attn(fig, axes[i, j], qs[i][j], 17, 15)

    fig.tight_layout()
    # fig.savefig(f'attn--{cmd_args.model_name}.pdf', format='pdf', dpi=800)
    return attentions


# %%
if __name__ == '__main__':
    # %%
    import pickle

    params = get_args()
    pl.seed_everything(params['seed'], workers=True)

    cmd_args = parser.parse_args([] if "__file__" not in globals() else None)

    torch.set_printoptions(linewidth=180, precision=2)
    np.set_printoptions(linewidth=180, precision=2)

    ckp_dir = os.path.join(
        params['logging']['ckp_dir'],
        f'{cmd_args.model_name}--{cmd_args.time_ckp}'
    )
    cfg_file = os.path.join(
        params['logging']['ckp_dir'].replace('checkpoints', ''),
        'slurm_logs',
        f'{cmd_args.model_name}__{cmd_args.time_cfg}',
        'backup',
        'cfg.yaml'
    )
    params = get_args(cfg_file)

    ckp = [f for f in os.listdir(ckp_dir) if 'f1' in f][0]
    ckp_path = os.path.join(ckp_dir, ckp)

    # %%
    train_len = lambda data: len(data.train_dataloader()) if params['debug']['limit_train_batches'] is None else params['debug']['limit_train_batches']
    params['batch_size'] = cmd_args.bs

    if cmd_args.mode == 'txt':
        dataset = DataPL(params)
        model = TextBasedModel.load_from_checkpoint(ckp_path, args=params, n_train_batches=train_len(dataset)).eval()

    else:
        w2v_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])
        dataset = AudioDataPL(params, w2v_model)
        model = FullModel.load_from_checkpoint(ckp_path, args=params, n_train_batches=train_len(dataset), audio_model=w2v_model).eval()

    # disable randomness, dropout, etc...
    # model.eval()

    attns = attention_matrices(model, dataset.val_dataloader(), dataset.dev_dataset)
    with open(f'attns-{cmd_args.model_name}.pickle', 'wb') as f:
        pickle.dump(attns, f, protocol=pickle.HIGHEST_PROTOCOL)

