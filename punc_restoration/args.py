import yaml
import argparse
import os
import collections
import torch
import math
from datetime import datetime


# parser = argparse.ArgumentParser()
# parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate the given model")
def get_args(path=None):
    if path is None:
        if 'lnet' in os.getcwd():
            config_path = '/lnet/express/work/people/stankov/alignment/Thesis/punc_restoration/cfg.yaml'
        else:
            config_path = 'cfg.yaml'
    else:
        config_path = path

    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    return args


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                v = ';'.join([f'{element}' for element in v])
            items.append((new_key, v))
    return dict(items)


def compute_receptive_field(strides, kernels, r=1, freq=16_000):
    j = 1
    for s, k in zip(strides, kernels):
        r = r + (k-1) * j
        j *= s
    return r / freq


def short_name_gen(params, cmd_args, ngpus=0):
    epochs = params['nn']['training_params']['epochs']
    lr = params['nn']['training_params']['lr']

    name = f'{cmd_args.mode}'

    if cmd_args.mode == 'sound':
        name += f'{params["mm_encoder"]["n_layers"]}'

    name += f'__ep{epochs}__lr{lr}__bs{cmd_args.bs}'

    if ngpus != 0:
        name += f'__ngpu_{ngpus}'
    elif torch.cuda.is_available():
        name += f'__ngpu_{torch.cuda.device_count()}'

    # add asr params
    if cmd_args.mode == 'sound':
        audio_params = params['audio_params']['cnn_model']
        strides = '.'.join([f"{s}" for s in audio_params['strides']])
        kernels = '.'.join([f"{k}" for k in audio_params['kernels']])
        dropouts = ''.join(['T' if d else 'F' for d in audio_params['use_do']])
        asr_params_str = '__str{strides}__ker{kernels}__b{bias}__do{dropout_rate:.1f}{dropouts}__frbckb{freeze_backbone}'.format(
            strides=strides,
            kernels=kernels,
            bias='T' if audio_params['bias'] else 'F',
            dropout_rate=audio_params['dropout'],
            dropouts=dropouts,
            freeze_backbone='T' if audio_params['freeze_backbone'] else 'F'
        )

        if audio_params['use_norm']:
            norm_dict = dict(
                LayerNorm='LN',
                BatchNorm='BN'
            )
            asr_params_str += '__{}'.format(norm_dict[audio_params['normalization']])

        asr_params_str += '__{}'.format(audio_params['activation'])

        if params['audio_params']['use_emb']:
            asr_params_str += '__embsize{embsize}__maxemb{maxemb}__embdo{embdo}'.format(
                embsize=params['audio_params']['emb']['embedding_dim'],
                maxemb=params['audio_params']['emb']['num_embeddings'],
                embdo=params['audio_params']['emb_do'],
            )

        if params['audio_params']['t_head']['use_transformer']:
            asr_params_str += '__layers{layers}__heads{heads}__ffdim{ffdim}'.format(
                layers=params['audio_params']['t_head']['layers'],
                heads=params['audio_params']['t_head']['nhead'],
                ffdim=params['audio_params']['t_head']['dim_feedforward']
            )

        if params['audio_params']['t_emb']['use_temb']:
            asr_params_str += '__tembn{tembn}'.format(
                tembn=params['audio_params']['t_emb']['num_embeddings']
            )

        if params['audio_params']['backbone'] == 'melspectrogram':
            mel_params = {**params['audio_params']['spec_args'], **params['audio_params']['mel_args']}
            mel_str = 'nfft{nfft}__hop{hop}__mls{mls}'.format(
                nfft=mel_params['n_fft'],
                hop=mel_params['hop_length'],
                mls=mel_params['n_mels']
            )
            asr_params_str += f'__{mel_str}'
            receptive_field = compute_receptive_field([mel_params['hop_length']] + audio_params['strides'], [mel_params['n_fft']] + audio_params['kernels'])
            asr_params_str += f'__rcpfld{receptive_field:.3f}'

        elif params['audio_params']['backbone'] == 'w2v2':
            asr_params_str += '__w2v2'
            receptive_field = compute_receptive_field([5, 2, 2, 2, 2, 2, 2] + audio_params['strides'], [10, 3, 3, 3, 3, 2, 2] + audio_params['kernels'])
            asr_params_str += f'__rcpfld{receptive_field:.3f}'

        elif params['audio_params']['backbone'] == 'baseline':
            asr_params_str += '__w2v2full'
            receptive_field = compute_receptive_field([5, 2, 2, 2, 2, 2, 2], [10, 3, 3, 3, 3, 2, 2])
            # in baseline we want only the receptive field
            asr_params_str = f'__rcpfld{receptive_field:.3f}'
        else:
            raise ValueError('Unknown value for `audio_params.backbone` = {}'.format(params['audio_params']['backbone']))

        if params['audio_params']['random_audio']:
            asr_params_str += '__RAND'

        name += asr_params_str

    name += f'__{cmd_args.name}'

    return name, datetime.now().strftime('%d.%m__%H.%M')


def dict_to_pretty_string(d):
    flat_d = flatten_dict(d)
    result = []
    for k, v in flat_d.items():
        result.append(f'- `{k.replace("_", "-")} = {v}`')
    return '\n'.join(result)
