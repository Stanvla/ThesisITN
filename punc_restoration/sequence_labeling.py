# %%
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

try:
    from args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from cls_head import ClassificationHead
    from load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL
    from multi_modal_att import MMEncoder
    from metrics import MetricsComp
    from audio_model import AudioEncoder
except ModuleNotFoundError as e:
    from punc_restoration.metrics import MetricsComp
    from punc_restoration.audio_model import AudioEncoder
    from punc_restoration.multi_modal_att import MMEncoder
    from punc_restoration.args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from punc_restoration.cls_head import ClassificationHead
    from punc_restoration.load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL

parser = argparse.ArgumentParser()
parser.add_argument("--name", default='dummy', type=str)
parser.add_argument("--mode", choices=['txt', 'sound'], type=str, required=True)
parser.add_argument("--bs", type=int, required=True)
parser.add_argument("--text_logger", default= '/lnet/express/work/people/stankov/alignment/Thesis/punc_restoration/debug_logger.log', type=str)
parser.add_argument("--enable_pb", default=True, type=bool)
parser.add_argument("--profiling", default=False, type=bool)


class TextBasedModel(pl.LightningModule):
    def __init__(self, args: dict, n_train_batches: int):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(args['robeczech_path'], revision='v1.0')
        self.freeze_emb()

        self.cls_punc = ClassificationHead(
            len(punc_labels),
            self.roberta.config.hidden_size,
            args['nn']['model_params']['cls_dropout']
        )
        self.cls_case = ClassificationHead(
            len(cap_labels),
            self.roberta.config.hidden_size,
            args['nn']['model_params']['cls_dropout']
        )
        self.train_params = args['nn']['training_params']
        self.total_steps = n_train_batches * args['nn']['training_params']['epochs']
        self.warmup_steps = int(self.total_steps * self.train_params['warmup_steps_perc'])
        self.log_steps = args['logging']['each_n_steps']
        self.n_worst_examples = args['logging']['n_worst_examples']
        self.params = args
        self.train_size = n_train_batches

    def freeze_emb(self):
        for p in self.roberta.embeddings.parameters():
            p.requires_grad = False

    def lr_scheduler(self, cur_step):
        scalar = (self.total_steps - cur_step) / (self.total_steps - self.warmup_steps)
        if cur_step < self.warmup_steps:
            scalar = cur_step / self.warmup_steps
        return float(scalar)

    def get_params(self, m):
        opt_parameters = []
        named_parameters = list(m.named_parameters())
        lr = self.train_params['lr']

        for layer in range(11, -1, -1):
            params = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n]
            layer_params = {"params": params, "lr": lr}
            opt_parameters.append(layer_params)

            lr *= self.train_params['lr_decay']

        return opt_parameters

    def get_head_params(self):
        return [
            {'params': list(self.cls_punc.parameters()), 'lr': self.train_params['lr']},
            {'params': list(self.cls_case.parameters()), 'lr': self.train_params['lr']},
        ]

    def configure_optimizers(self):
        opt_params = self.get_params(self.roberta)
        opt_params.extend(self.get_head_params())

        optimizer = optim.Adam(opt_params, lr=self.train_params['lr'])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: self.lr_scheduler(step))
        # https://github.com/Lightning-AI/lightning/issues/328#issuecomment-717085681
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html?highlight=learning%20rate%20scheduler#bring-your-own-custom-learning-rate-schedulers
        return [optimizer], dict(scheduler=scheduler, interval='step', frequency=1)

    # def forward(self, batch, mask):
    def forward(self, batch):
        text_ids, text_mask = batch['text_ids'], batch['masks']
        outputs = self.roberta(text_ids, attention_mask=text_mask, output_attentions=True)
        # named_tuple with
        #   `last_hidden_state` of shape torch.Size([batch_size, max_seq_len, hid_dim])
        #   `pooler_output` of shape torch.Size([batch_size, hid_dim])

        outputs_punc = self.cls_punc(outputs.last_hidden_state)
        outputs_case = self.cls_case(outputs.last_hidden_state)
        # outputs_{punc,case}.shape: torch.Size([batch_size, max_seq_len, n_classes])

        return outputs_punc, outputs_case, outputs.attentions[-1], None

    def get_loss_metrics(self, logits_batch, trgs_batch):
        # loss and metrics are averaged

        # https://huggingface.co/docs/transformers/tasks/token_classification#evaluate
        # if we do not use IGNORE_INDEX, then we will need to iterate over the batch and compute loss for each
        # sequence because each sequence has different length
        batch_size, max_seq_len, n_classes = logits_batch.shape

        losses = []
        for logits, trg in zip(logits_batch, trgs_batch):
            loss = F.cross_entropy(logits, trg).detach()
            losses.append(loss)

        # total_loss /= batch_size

        total_loss = F.cross_entropy(
            logits_batch.reshape(batch_size * max_seq_len, n_classes),
            trgs_batch.reshape(batch_size * max_seq_len),
            reduction='mean'
        )
        predictions_batch = logits_batch.argmax(2)
        # acc = torch.sum(trgs_batch[trgs_batch != IGNORE_INDEX] == predictions_batch[trgs_batch != IGNORE_INDEX])
        not_ignored_cnt = torch.sum(trgs_batch != IGNORE_INDEX)

        # return loss, acc.detach() / not_ignored_cnt, predictions_batch, not_ignored_cnt
        return total_loss, predictions_batch.detach(), not_ignored_cnt.detach(), losses

    def _step(self, batch):
        # inps, punc_trgs, cap_trgs, masks, seq_lens = batch['text_ids'], batch['punc_trgs'], batch['cap_trgs'], batch['masks'], batch['seq_lens']
        # outputs_punc, outputs_case = self(inps, masks)
        # shape = [batch_size, max_seq_len, n_classes]

        punc_trgs, cap_trgs, seq_lens = batch['punc_trgs'], batch['cap_trgs'], batch['seq_lens']
        outputs_punc, outputs_case, text_attn, cross_attn = self(batch)
        # shape = [batch_size, max_seq_len, n_classes]

        if self.training:
            outputs_punc = outputs_punc / self.train_params['softmax_temp']
            outputs_case = outputs_case / self.train_params['softmax_temp']

        punc_loss, punc_preds, not_ignored_cnt, punc_losses = self.get_loss_metrics(outputs_punc, punc_trgs)
        case_loss, case_preds, _, case_losses = self.get_loss_metrics(outputs_case, cap_trgs)

        results = dict(
            loss=(punc_loss + case_loss) / 2,
            punc_loss=punc_loss,
            case_loss=case_loss,
            # batch_size=not_ignored_cnt,
        )
        return results, punc_preds, case_preds, punc_losses, case_losses, text_attn, cross_attn

    def training_step(self, batch, idx):
        results, *_ = self._step(batch)
        return results

    def validation_step(self, batch, batch_idx):
        # punc_loss, punc_acc, batch_size, punc_preds = self._step(batch)
        results, punc_preds, case_preds, punc_losses, case_losses, text_attn, cross_attn = self._step(batch)

        # `verbose` will be used to compute different metrics
        results['verbose'] = dict(
            indices=batch['indices'],
            max_len=batch['max_seq_len'],
            losses=torch.tensor([(punc_loss + case_loss) / 2 for punc_loss, case_loss in zip(punc_losses, case_losses)]),
            punc_trgs=batch['punc_trgs'].detach(),
            case_trgs=batch['cap_trgs'].detach(),
            seq_lens=batch['seq_lens'],
            punc_preds=punc_preds,
            case_preds=case_preds,
            text_attn=torch.mean(text_attn.detach(), dim=1),
            cross_attn=cross_attn,
        )
        results['loss'] = results['loss'].detach()
        return results

    def training_step_end(self, train_step_outputs):
        if self.global_step % self.log_steps == 0 and self.global_rank == 0:
            # logging.info(f'step{self.global_step:8}')
            self.logger.experiment.add_scalar(f'Loss/Train', train_step_outputs['loss'], self.global_step)
            self.logger.experiment.add_scalar(f'LossPunc/Train', train_step_outputs['punc_loss'], self.global_step)
            self.logger.experiment.add_scalar(f'LossCase/Train', train_step_outputs['case_loss'], self.global_step)

        return train_step_outputs

    def validation_epoch_end(self, outputs):
        metric_comp = MetricsComp(outputs, dataset.dev_dataset)

        max_len = metric_comp.get_max_len()
        max_len = torch.max(self.all_gather(max_len))

        max_audio_len = None
        if outputs[0]['verbose']['cross_attn'] is not None:
            max_audio_len = metric_comp.get_max_audio_len()
            max_audio_len = torch.max(self.all_gather(max_audio_len))

        agg = metric_comp.aggregate_all(max_len, max_audio_len)

        metrics = {k: v for k, v in agg.items() if k != 'verbose'}
        metrics = self.all_gather(metrics)
        metrics = {k: v.mean().item() for k, v in metrics.items()}

        agg['verbose'] = self.all_gather(agg['verbose'])
        verbose = agg['verbose']
        # now need to reduce "gpu_batch" dimension
        verbose = {k: torch.cat([x for x in v]) for k, v in verbose.items()}

        # before gathering
        # indices = tensor([ 0,  2,  4,  6,  8, 10, 12, 14])
        # after gathering
        # indices = tensor([[ 0,  2,  4,  6,  8, 10, 12, 14],
        #                   [ 1,  3,  5,  7,  9, 11, 13, 15]], device='cuda:0')
        # the same for the other values in verbose, outputs from different gpus are stacked in a large batch
        # (adding new "gpu_batch" dimension -> batch over batches)

        # multiple lines on one plot
        # https://stackoverflow.com/questions/73399268/how-to-plot-multiple-scalars-in-tensorboard-in-the-same-figure-without-spamming
        if self.global_rank == 0:
            # logging.info(f'step{self.global_step:8} validation')
            # log loss
            self.logger.experiment.add_scalar(f'Loss/Val', metrics['loss'], self.global_step)
            self.logger.experiment.add_scalar(f'LossPunc/Val', metrics['punc_loss'], self.global_step)
            self.logger.experiment.add_scalar(f'LossCase/Val', metrics['case_loss'], self.global_step)

            # log text
            self.logger.experiment.add_text('BadExamples', metric_comp.worst_examples_md(self.n_worst_examples, verbose), self.global_step)

            # log confusion matrices
            cm_punc_img, cm_punc_md = metric_comp.confusion_matrix(verbose, 'punc_preds', 'punc_trgs', punc_labels, int2punc_labels)
            cm_case_img, cm_case_md = metric_comp.confusion_matrix(verbose, 'case_preds', 'case_trgs', cap_labels, int2cap_labels)
            self.logger.experiment.add_figure('ConfMatrixPunc/Val', cm_punc_img, self.global_step)
            self.logger.experiment.add_figure('ConfMatrixCase/Val', cm_case_img, self.global_step)
            self.logger.experiment.add_text('ConfMatrixPunc/Val', cm_punc_md, self.global_step)
            self.logger.experiment.add_text('ConfMatrixCase/Val', cm_case_md, self.global_step)

            # log any punc precision and recall
            # pr, rec = metric_comp.any_punc_precision_recall(verbose)
            # self.logger.experiment.add_scalar(f'AnyPuncPrecision/Val', pr, self.global_step)
            # self.logger.experiment.add_scalar(f'AnyPuncRecall/Val', rec, self.global_step)

            first_q_prec, first_q_rec, last_q_prec, last_q_rec, prec, req = metric_comp.any_punc_quantiles(verbose)
            self.logger.experiment.add_scalar('AnyPuncPrecByLen/ShortQ1', first_q_prec, self.global_step)
            self.logger.experiment.add_scalar('AnyPuncPrecByLen/LongQ4', last_q_prec, self.global_step)
            self.logger.experiment.add_scalar('AnyPuncPrecByLen/All', prec, self.global_step)
            self.logger.experiment.add_scalar('AnyPuncRecallByLen/ShortQ1', first_q_rec, self.global_step)
            self.logger.experiment.add_scalar('AnyPuncRecallByLen/LongQ4', last_q_rec, self.global_step)
            self.logger.experiment.add_scalar('AnyPuncRecallByLen/All', req, self.global_step)

            # f1 scores
            punc_total_f1, punc_f1_md = metric_comp.f1_scores(verbose, 'punc_preds', 'punc_trgs', punc_labels, int2punc_labels)
            case_total_f1, case_f1_md = metric_comp.f1_scores(verbose, 'case_preds', 'case_trgs', cap_labels, int2cap_labels)
            self.logger.experiment.add_scalar('F1/Val', (punc_total_f1 + case_total_f1) / 2, self.global_step)
            self.logger.experiment.add_scalar('WeightedF1Punc/Val', punc_total_f1, self.global_step)
            self.logger.experiment.add_scalar('WeightedF1Case/Val', case_total_f1, self.global_step)
            self.logger.experiment.add_text('F1Punc/Val', punc_f1_md, self.global_step)
            self.logger.experiment.add_text('F1Case/Val', case_f1_md, self.global_step)

            # log hp-params
            self.log("hp/loss", metrics['loss'])

            # log text_attn
            text_attn = metric_comp.attention_vis(verbose, 'text_attn')
            self.logger.experiment.add_figure('Text Encoder Self Attention', text_attn, self.global_step)

            # log cross attentions if presented
            cross_attentions = metric_comp.attention_vis(verbose, 'cross_attn')
            if cross_attentions is not None:
                self.logger.experiment.add_figure('Cross Attention', cross_attentions, self.global_step)

            plt.close(cm_punc_img)
            plt.close(cm_case_img)

        punc_total_f1, _ = metric_comp.f1_scores(verbose, 'punc_preds', 'punc_trgs', punc_labels, int2punc_labels)
        case_total_f1, _ = metric_comp.f1_scores(verbose, 'case_preds', 'case_trgs', cap_labels, int2cap_labels)

        # checkpointing
        self.log('f1', (punc_total_f1 + case_total_f1) / 2, logger=False, sync_dist=False, rank_zero_only=True)
        self.log('global_step', self.global_step, logger=False, sync_dist=False, rank_zero_only=True)
    def on_train_start(self):

        layout = {
            "MultiScalars": {
                'AnyPuncPrecByLen': ['Multiline', ['AnyPuncPrecByLen/ShortQ1', 'AnyPuncPrecByLen/LongQ4', 'AnyPuncPrecByLen/All']],
                'AnyPuncRecallByLen': ['Multiline', ['AnyPuncRecallByLen/ShortQ1', 'AnyPuncRecallByLen/LongQ4', 'AnyPuncRecallByLen/All']]
            }
        }
        self.logger.experiment.add_custom_scalars(layout)

        params = flatten_dict(self.params['nn'])
        params['batch_size'] = self.params['batch_size']
        params['train_size'] = self.train_size

        self.logger.log_hyperparams(params, {'hp/loss': 0})


class FullModel(TextBasedModel):
    def __init__(self, args, n_train_batches, audio_model):
        super().__init__(args, n_train_batches)
        self.use_audio_baseline = args['audio_params']['backbone'] == 'baseline'

        if self.use_audio_baseline:
            self.audio_model = audio_model
            self.audio_model.config.mask_time_prob = 0
        else:
            self.audio_model = AudioEncoder(args, audio_model)

        self.freeze_backbone = args['audio_params']['cnn_model']['freeze_backbone']

        self.pos_emb = None
        self.pos_emb_do = None
        if args['audio_params']['use_emb'] and not self.use_audio_baseline:
            self.pos_emb = nn.Embedding(**args['audio_params']['emb'])
            self.cnt_emb = nn.Embedding(embedding_dim=args['audio_params']['emb']['embedding_dim'], num_embeddings=768)
            self.pos_emb_do = nn.Dropout(args['audio_params']['emb_do'])

        if self.freeze_backbone and not self.use_audio_baseline:
            self.audio_model.freeze_backbone()
        self.mm_encoder = MMEncoder(args, self.roberta.config)

    def configure_optimizers(self):
        opt, scheduler_dict = super().configure_optimizers()

        # special case for audio baseline
        if self.use_audio_baseline:
            for p in self.audio_model.parameters():
                opt[0].add_param_group({'params': p, 'lr': self.train_params['lr']})
            return opt, scheduler_dict

        for p in self.audio_model.conv_layers.parameters():
            opt[0].add_param_group({'params': p, 'lr': self.train_params['lr']})

        if not self.freeze_backbone:
            opt[0].add_param_group({'params': self.audio_model.backbone.feature_extractor.parameters(), 'lr': self.train_params['lr'] * self.train_params['lr_decay']})

        opt[0].add_param_group(
            {'params': list(self.mm_encoder.parameters()), 'lr': self.train_params['lr']},
        )
        # opt = optim.Adam(self.parameters(), lr=self.train_params['lr'])
        return opt, scheduler_dict

    def forward(self, batch):
        # batch['positional_ids_txt'] = [[0, 1, 1, 2, 2, 3, 4, 4, 4, ..., 511, 511], ...] -> index repeats, meaning that the word is split into multiple tokens
        # batch['positional_ids_audio'] = [[0, 1, 1, 3, 4, 4, 6, ..., 511, 511], ...] -> as above, but some indices might be skipped, since the sound overlays
        # 511 in both ids are for padding

        text_ids, sound_inp, text_mask, sound_raw_len, sound_f_len = batch['text_ids'], batch['audio'], batch['masks'], batch['raw_len'], batch['feature_len']
        positional_ids_txt, positional_ids_audio, positional_ids_audio_cnt = batch['positional_ids_txt'], batch['positional_ids_audio'], batch['positional_ids_audio_cnt']

        text_outs = self.roberta(text_ids, attention_mask=text_mask, output_attentions=True)
        text_h = text_outs.last_hidden_state

        if self.pos_emb is not None:
            positional_ids_txt = self.pos_emb(positional_ids_txt)

            positional_ids_audio = self.pos_emb(positional_ids_audio) + self.cnt_emb(positional_ids_audio_cnt)

            text_h = self.pos_emb_do(text_h + positional_ids_txt)
        else:
            positional_ids_audio = None

        if self.use_audio_baseline:
            audio_h = self.audio_model(sound_inp)
            audio_h = audio_h.last_hidden_state
        else:
            audio_h = self.audio_model(sound_inp, pos_emb=positional_ids_audio, f_len=sound_f_len, device=self.device)


        # ic(audio_h.shape)
        # ic(text_h.shape)
        new_txt_mask = text_mask == 0

        mm_output, attention_weights = self.mm_encoder(text_h, audio_h, text_padding_mask=new_txt_mask)

        outputs_punc = self.cls_punc(mm_output)
        outputs_case = self.cls_case(mm_output)

        return outputs_punc, outputs_case, text_outs.attentions[-1], attention_weights

    def validation_step(self, batch, batch_idx):
        result = super().validation_step(batch, batch_idx)
        result['verbose']['cross_attn'] = result['verbose']['cross_attn'].detach()
        return result

    def on_train_start(self):

        layout = {
            "MultiScalars": {
                'AnyPuncPrecByLen': ['Multiline', ['AnyPuncPrecByLen/ShortQ1', 'AnyPuncPrecByLen/LongQ4', 'AnyPuncPrecByLen/All']],
                'AnyPuncRecallByLen': ['Multiline', ['AnyPuncRecallByLen/ShortQ1', 'AnyPuncRecallByLen/LongQ4', 'AnyPuncRecallByLen/All']]
            }
        }
        self.logger.experiment.add_custom_scalars(layout)

        params = flatten_dict(self.params['nn'])
        audio_params = flatten_dict(self.params['audio_params']['cnn_model'])
        audio_t_prams = flatten_dict(self.params['audio_params']['t_head'])
        audio_mfcc_params = {**flatten_dict(self.params['audio_params']['spec_args']), **flatten_dict(self.params['audio_params']['mel_args'])}

        params['batch_size'] = self.params['batch_size']
        params['train_size'] = self.train_size

        self.logger.log_hyperparams({**params, **audio_params, **audio_mfcc_params, **audio_t_prams}, {'hp/loss': 0})

    # logging model weights
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md

# %%
if __name__ == '__main__':
    # %%
    params = get_args()
    pl.seed_everything(params['seed'], workers=True)

    cmd_args = parser.parse_args([] if "__file__" not in globals() else None)
    params['batch_size'] = cmd_args.bs

    torch.set_printoptions(linewidth=180, precision=2)
    np.set_printoptions(linewidth=180, precision=2)

    train_len = lambda data: len(data.train_dataloader()) if params['debug']['limit_train_batches'] is None else params['debug']['limit_train_batches']

    if cmd_args.mode == 'txt':
        dataset = DataPL(params)
        model = TextBasedModel(params, train_len(dataset))
    else:
        w2v_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])
        dataset = AudioDataPL(params, w2v_model)
        model = FullModel(params, train_len(dataset), w2v_model)

    short_name, cur_time = short_name_gen(params, cmd_args)
    logger = TensorBoardLogger(save_dir=params['logging']['log_dir'], name=short_name, version=cur_time)

    callbacks = None
    if params['logging']['enable_ckp']:
        ckp_path = os.path.join(
            params['logging']['ckp_dir'],
            f'{short_name}--{cur_time}'
        )

        # save based on valid loss and valid acc
        f1_checkpoint_callback = ModelCheckpoint(
            monitor="f1",
            mode='max',
            dirpath=ckp_path,
            filename='step_{step:06d}__f1_{f1:.3f}',
            save_weights_only=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="global_step",
            mode='max',
            dirpath=ckp_path,
            filename="step_{step:06d}",
            save_weights_only=True,
        )
        callbacks = [f1_checkpoint_callback, checkpoint_callback]

    trainer_params = dict(
        num_sanity_val_steps=0,
        deterministic=True,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        precision=16 if torch.cuda.is_available() else 32,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=params['logging']['enable_ckp'],
        enable_progress_bar=cmd_args.enable_pb,
        max_epochs=params['nn']['training_params']['epochs'],
        limit_train_batches=params['debug']['limit_train_batches'],
        limit_val_batches=params['debug']['limit_val_batches'],
        val_check_interval=params['logging']['val_check_interval'],
    )

    if torch.cuda.device_count() > 1:
        trainer_params['strategy'] = 'ddp'

    if cmd_args.profiling:
        from pytorch_lightning.profiler import PyTorchProfiler
        trainer_params['profiler'] = PyTorchProfiler()
    # trainer_params['accelerator'] = 'cpu'
    # trainer_params['strategy'] = None
    # trainer_params['precision'] = 32
    # trainer_params['devices'] = None

    trainer = pl.Trainer(**trainer_params)

    trainer.fit(model, dataset)
    # %%
    # %%
