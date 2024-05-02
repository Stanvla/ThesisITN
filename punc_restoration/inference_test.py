import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
import logging
from icecream import ic
import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, Wav2Vec2Model
from sklearn.metrics import f1_score

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

def get_params(model_name, ckpt_time, cfg_time):
    ckp_dir = os.path.join(
        ckpt_dir,
        f'{model_name}--{ckpt_time}'
    )

    cfg_file = os.path.join(
        ckpt_dir.replace('checkpoints', ''),
        'slurm_logs',
        f'{model_name}__{cfg_time}',
        'backup',
        'cfg.yaml'
    )
    params = get_args(cfg_file)

    ckp = [f for f in os.listdir(ckp_dir) if 'f1' in f][0]
    ckp_path = os.path.join(ckp_dir, ckp)

    return params, ckp_path


def get_train_len(data):
    return len(data.train_dataloader()) if params['debug']['limit_train_batches'] is None else params['debug']['limit_train_batches']


def load_model_and_data(mode, ckpt_path, params, bs):
    params['batch_size'] = bs
    if mode == 'txt':
        dataset = DataPL(params)
        model = TextBasedModel.load_from_checkpoint(ckpt_path, args=params, n_train_batches=get_train_len(dataset))
    else:
        w2v_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])
        dataset = AudioDataPL(params, w2v_model)
        model = FullModel.load_from_checkpoint(ckpt_path, args=params, n_train_batches=get_train_len(dataset), audio_model=w2v_model)
    model.eval()
    return model, dataset


def get_pred_text(token_indices, norm_text, case_preds, punc_preds):
    prev_token_id = None
    aligned_pred = []

    for token_id, p_pred, c_pred in zip(token_indices, punc_preds, case_preds):
        if token_id is None:
            continue
        if prev_token_id != token_id:
            aligned_pred.append((
                int2punc_labels[p_pred], int2cap_labels[c_pred]
            ))
        prev_token_id = token_id

    # apply labels
    labeled_text = []
    for w, (punc_label, case_label) in zip(norm_text.split(), aligned_pred):
        new_w = w
        if case_label == '[cap]':
            new_w = new_w.title()
        if case_label == '[all_cap]':
            new_w = new_w.upper()

        if punc_label != BLANK:
            new_w += punc_label

        labeled_text.append(new_w)

    return ' '.join(labeled_text)


def get_predictions(model, loader, data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = model.to(device)
    predictions_texts = []
    outputs = dict(
        data_indices=[],
        norm_texts=[],
        gold_texts=[],
        # token_indices=[],

        case_preds=[],
        case_trgs=[],

        punc_preds=[],
        punc_trgs=[],

        punc_losses=[],
        case_losses=[],
    )

    for i, batch in enumerate(loader):
        if cmd_args.debug and i == 100:
            break
        if i % 100 == 0:
            print(f'Batch {i:4}')
        batch_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        punc_trgs, case_trgs, seq_lens, indices = batch['punc_trgs'], batch['cap_trgs'], batch['seq_lens'], batch['indices']

        with torch.no_grad():
            outputs_punc, outputs_case, text_attn, cross_attn = trained_model(batch_gpu)
            punc_loss, punc_preds, _, punc_losses = trained_model.get_loss_metrics(outputs_punc.to(device), punc_trgs.to(device))
            case_loss, case_preds, _, case_losses = trained_model.get_loss_metrics(outputs_case.to(device), case_trgs.to(device))
        outputs['data_indices'].extend(batch['indices'])

        outputs['case_preds'].extend([p for p in case_preds.cpu().numpy().tolist()])
        outputs['case_trgs'].extend([t for t in case_trgs.cpu().numpy().tolist()])

        outputs[f'punc_preds'].extend([p for p in punc_preds.cpu().numpy().tolist()])
        outputs['punc_trgs'].extend([t for t in punc_trgs.cpu().numpy().tolist()])

        outputs['punc_losses'].extend([p.item() for p in punc_losses])
        outputs['case_losses'].extend([c.item() for c in case_losses])

        for k, i in enumerate(indices):
            verbose = data.get_verbose(i)
            outputs['norm_texts'].append(verbose['norm_text'])
            outputs['gold_texts'].append(verbose['gold_text'])

            predictions_texts.append(get_pred_text(
                verbose['word_ids'],
                verbose['norm_text'],
                case_preds.cpu().numpy().tolist()[k],
                punc_preds.cpu().numpy().tolist()[k]
            ))

    return pd.DataFrame(outputs), predictions_texts

def get_f1_scores(result_df, pred_col, trg_col, int2label, labels):
    preds = [
        int2label[p]
        for l, lt in zip(result_df[pred_col].tolist(), result_df[trg_col].tolist())
        for i, p in enumerate(l) if lt[i] != IGNORE_INDEX
    ]

    trgs = [int2label[t] for l in result_df[trg_col].tolist() for t in l if t != IGNORE_INDEX]
    scores = f1_score(trgs, preds, average=None, labels=labels, zero_division=0)
    return {l:score for l, score in zip(labels, scores)}


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='sound')
# parser.add_argument("--bs", type=int, default=200)
# parser.add_argument('--model_name', type=str, default=)
# parser.add_argument("--time_ckp", type=str, default=)
# parser.add_argument("--time_cfg", type=str, default=)
parser.add_argument("--model_type", type=str, default='last')
parser.add_argument("--debug", type=bool, default=False)


# %%
if __name__ == '__main__':
    # %%
    off_names = [
        # 'text_only',
        # 'audio_baseline',
        # 'mfcc',
        # 'embeddings',
        # 'random-sound',
    ]
    model_names = [
    ]

    ckpt_times = [
    ]
    cfg_times = [
    ]

    modes = [
    ]

    batch_sizes = [
        # 200,
        # 8,
        # 200,
        # 200,
        # 200,
    ]

    seed = 0xDEAD
    ckpt_dir = '/lnet/express/work/people/stankov/alignment/Thesis/punc_restoration/checkpoints'

    torch.set_printoptions(linewidth=180, precision=2)
    np.set_printoptions(linewidth=180, precision=2)
    pl.seed_everything(seed, workers=True)

    cmd_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_scores_punc_all = []
    f1_scores_case_all = []
    results_df = None

    for model_name, off_name, ckpt_time, cfg_time, mode, bs in zip(model_names, off_names, ckpt_times, cfg_times, modes, batch_sizes):
        print(off_name)
        if cmd_args.debug:
            bs = max(bs // 10, 2)
        params, ckpt_path = get_params(model_name, ckpt_time, cfg_time)
        model, dataset = load_model_and_data(mode, ckpt_path, params, bs)
        tmp_results_df, predictions = get_predictions(model, dataset.test_dataloader(), dataset.test_dataset)
        tmp_results_df.to_parquet(f'{off_name}--predictions-test.parquet')

        # f1_scores_punc = get_f1_scores(tmp_results_df, 'punc_preds', 'punc_trgs', int2punc_labels, punc_labels)
        # f1_scores_case = get_f1_scores(tmp_results_df, 'case_preds', 'case_trgs', int2cap_labels, cap_labels)
        # f1_scores_punc_all.append(f1_scores_punc)
        # f1_scores_case_all.append(f1_scores_case)

        # if results_df is None:
        #     results_df = tmp_results_df
        # results_df[off_name] = predictions

    # f1_punc_df = pd.DataFrame(f1_scores_punc_all)
    # f1_case_df = pd.DataFrame(f1_scores_case_all)
    # f1_punc_df.index = off_names
    # f1_case_df.index = off_names
    #
    # results_df.to_parquet('texts-test.parquet')
    # f1_punc_df.to_parquet('f1-punc-test.parquet')
    # f1_case_df.to_parquet('f1-case-test.parquet')
    # %%
    # %%

