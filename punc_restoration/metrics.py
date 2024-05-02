import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import math

try:
    from args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL
except ModuleNotFoundError as e:
    from punc_restoration.args import get_args, dict_to_pretty_string, flatten_dict, short_name_gen
    from punc_restoration.load_data import DataPL, punc_labels, int2punc_labels, int2cap_labels, cap_labels, IGNORE_INDEX, BLANK, AudioDataPL


class MetricsComp:
    def __init__(self, outputs, dataset):
        if isinstance(outputs, dict):
            outputs = [outputs]
        self.outputs = outputs
        self.dataset = dataset

    def get_max_len(self):
        return max([o['verbose']['max_len'] for o in self.outputs])

    def get_max_audio_len(self):
        return max([o['verbose']['cross_attn'].shape[-1] for o in self.outputs])

    def aggregate_verbose_field(self, field, max_seq_len):
        return torch.cat([
            F.pad(
                o['verbose'][field],
                (0, max_seq_len - o['verbose'][field].shape[1]),
                value=IGNORE_INDEX
            )
            for o in self.outputs
        ])

    def aggregate_verbose(self, max_seq_len, max_audio_len):
        # self.outputs is a list of batches (of length N)
        # take field `case_trgs`, we will have list of batches, where self.outputs[i]['verbose']['case_trgs'] is a batch of case targets
        # after the aggregation new_outputs will no longer be a list, but a dict
        # and new_outputs['verbose']['case_trgs'] will be a list of size (N * self.outputs[i]['verbose']['case_trgs'].shape[0])
        # the size is given by the double for cycle

        if 'verbose' not in self.outputs[0]:
            raise ValueError(f'Can not aggregate verbose output, `verbose` is not in the dict')

        verbose = {}

        verbose['punc_trgs'] = self.aggregate_verbose_field('punc_trgs', max_seq_len)
        verbose['case_trgs'] = self.aggregate_verbose_field('case_trgs', max_seq_len)

        verbose['punc_preds'] = self.aggregate_verbose_field('punc_preds', max_seq_len)
        verbose['case_preds'] = self.aggregate_verbose_field('case_preds', max_seq_len)

        verbose['losses'] = torch.tensor([l for o in self.outputs for l in o['verbose']['losses']])
        verbose['indices'] = torch.tensor([idx for o in self.outputs for idx in o['verbose']['indices']])
        verbose['seq_lens'] = torch.tensor([idx for o in self.outputs for idx in o['verbose']['seq_lens']])

        attn_pad = lambda attn: (0, max_seq_len - attn.shape[-1], 0, max_seq_len - attn.shape[1])
        verbose['text_attn'] = torch.cat([
            F.pad(o['verbose']['text_attn'], attn_pad(o['verbose']['text_attn']), value=0)
            for o in self.outputs
        ])

        if self.outputs[0]['verbose']['cross_attn'] is not None:
            cross_attn_pad = lambda attn: (0, max_audio_len - attn.shape[-1], 0, max_seq_len - attn.shape[1])
            verbose['cross_attn'] = torch.cat([
                F.pad(o['verbose']['cross_attn'], cross_attn_pad(o['verbose']['cross_attn']), value=0)
                for o in self.outputs
            ])

        return verbose

    def aggregate_num(self):
        aggregated = {k: 0 for k in self.outputs[0].keys() if k != 'verbose'}
        for k in self.outputs[0].keys():
            if k == 'verbose':
                continue
            for o in self.outputs:
                aggregated[k] += o[k]
            if k != 'batch_size':
                aggregated[k] /= len(self.outputs)
        return aggregated

    def aggregate_all(self, max_seq_len, max_audio_len):
        agg = self.aggregate_num()
        if 'verbose' in self.outputs[0]:
            agg['verbose'] = self.aggregate_verbose(max_seq_len, max_audio_len)
        return agg

    def add_labeled_text(self, batch_verbose, idx, norm_text, word_ids):
        prev_word_id = None
        aligned_pred = []

        for word_id, p_pred, c_pred in zip(word_ids, batch_verbose['punc_preds'][idx], batch_verbose['case_preds'][idx]):
            if word_id is None:
                continue
            if prev_word_id != word_id:
                aligned_pred.append((
                    int2punc_labels[p_pred.item()], int2cap_labels[c_pred.item()]
                ))
            prev_word_id = word_id

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

    def worst_examples(self, n_worst, outputs_verbose, comp_func=None):
        if comp_func is None:
            comp_func = lambda new, curr: new > curr

        # worst[i] = [loss, dataset_idx, outputs_idx]
        worst = []
        used_indices = set()
        losses = outputs_verbose['losses'].cpu().tolist()
        for _ in range(n_worst):
            candidate_loss = 0
            candidate_output_idx = 0
            indices = sorted(set(range(len(losses))) - used_indices)

            for i in indices:
                if comp_func(losses[i], candidate_loss):
                # if losses[i] > candidate_loss:
                    candidate_loss = losses[i]
                    candidate_output_idx = i

            worst.append([
                candidate_loss, outputs_verbose['indices'][candidate_output_idx].item(), candidate_output_idx
            ])
            used_indices.add(candidate_output_idx)
        return worst

    def worst_examples_md(self, n_worst, outputs_verbose):
        worst = self.worst_examples(n_worst, outputs_verbose)

        # examples[i] = [loss, labeled_text, gold_text]
        examples = []
        for loss, dataset_idx, out_idx in worst:
            data = self.dataset.get_verbose(dataset_idx)
            examples.append([
                loss,
                self.add_labeled_text(outputs_verbose, out_idx, data['norm_text'], data['word_ids']),
                data['gold_text']
            ])

        return self.get_markdown(examples)

    def get_markdown(self, lst):
        delimiter = lambda x: '\n' + '-' * x + '\n'
        result = []
        for loss, pred, gold in lst:
            result.append(
                f'- Loss = `{loss}`\n'
                f'- Pred = {pred}\n'
                f'- Gold = {gold}'
            )
        return delimiter(20).join(result)

    def verbose_prep(self, batch_verbose, field_pred, field_trg, indices=None):
        if indices is None:
            indices = torch.arange(batch_verbose[field_pred].shape[0])

        preds = batch_verbose[field_pred][indices]
        trgs = batch_verbose[field_trg][indices]

        preds = preds[trgs != IGNORE_INDEX].reshape(-1).cpu()
        trgs = trgs[trgs != IGNORE_INDEX].reshape(-1).cpu()

        return preds, trgs

    def confusion_matrix(self, batch_verbose, field_pred, field_trg, labels, int2labels):
        plt.rcParams["figure.figsize"] = (10, 10)
        # returns figure and markdown
        preds, trgs = self.verbose_prep(batch_verbose, field_pred, field_trg)

        preds = [int2labels[p] for p in preds.tolist()]
        trgs = [int2labels[p] for p in trgs.tolist()]

        disp = ConfusionMatrixDisplay.from_predictions(trgs, preds, normalize='true', cmap=plt.cm.Blues, labels=labels)

        cm = confusion_matrix(trgs, preds, normalize='true', labels=labels)
        df = pd.DataFrame(data=np.around(cm, 3), index=labels, columns=labels)

        return disp.figure_, df.to_markdown()

    def any_punc_precision_recall(self, batch_verbose, indices=None):
        # how well model can detect punctuation without taking into account punc labels

        preds, trgs = self.verbose_prep(batch_verbose, 'punc_preds', 'punc_trgs', indices)

        preds = ((preds > 0) + 0).tolist()
        trgs = ((trgs > 0) + 0).tolist()

        precision = precision_score(trgs, preds)
        recall = recall_score(trgs, preds)

        return precision, recall

    def f1_scores(self, batch_verbose, field_pred, field_trg, labels, int2labels):
        # returns weighted avg of all f1-scores and a separate score for each label in form of a markdown
        preds, trgs = self.verbose_prep(batch_verbose, field_pred, field_trg)

        preds = [int2labels[p] for p in preds.tolist()]
        trgs = [int2labels[p] for p in trgs.tolist()]

        # weighted_avg = f1_score(trgs, preds, average='macro', zero_division=0)

        scores = f1_score(trgs, preds, average=None, labels=labels, zero_division=0)
        # avg_score = np.average(scores)
        df = pd.DataFrame(data=np.expand_dims(scores, axis=0), columns=labels)

        avg_columns = [c for c in df.columns if c != 'weighted_avg' and c != '[blank]']
        non_blank_avg = df[avg_columns].iloc[0].mean(axis=0)

        df['weighted_avg'] = non_blank_avg

        return non_blank_avg, df.to_markdown()

    def any_punc_quantiles(self, batch_verbose):
        sorted_lens, indices = torch.sort(batch_verbose['seq_lens'])
        first_q = indices[: math.ceil(indices.shape[0] * 0.25)]
        last_q = indices[math.ceil(indices.shape[0] * 0.75):]

        first_q_prec, first_q_rec = self.any_punc_precision_recall(batch_verbose, first_q)
        last_q_prec, last_q_rec = self.any_punc_precision_recall(batch_verbose, last_q)
        prec, req = self.any_punc_precision_recall(batch_verbose)

        return first_q_prec, first_q_rec, last_q_prec, last_q_rec, prec, req

    def audio_ticks_visuzalize_idx(self, data):
        ends = data['feature_ends']
        audio_len = ends[-1]
        audio_ticks = [''] * (audio_len + 1)
        for end, word in zip(ends, data['gold_text'].split()):
            audio_ticks[end] = word
        return audio_ticks, audio_len

    def text_ticks_visualize_idx(self, data):
        # create text labels
        text_ticks = []
        prev_word_id = None
        gold_text_idx = 0
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

        return text_ticks

    def attention_visualize_ax(self, ax, attention, x_ticks, y_ticks, x_label, y_label, title):
        cax = ax.matshow(attention, cmap='bone', aspect='auto')
        # fig.colorbar(cax)

        # Set up axes
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(labels=x_ticks, rotation=40)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_xlabel(x_label)

        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(labels=y_ticks)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        return cax

    def cross_attention_visualize_idx(self, batch_verb, idx, ax, title, attn_name):
        # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#visualizing-attention
        dataset_idx = batch_verb['indices'][idx].item()
        data = self.dataset.get_verbose(dataset_idx)

        # create audio labels
        # for each word in sound we have an index where it ends in the feature space
        audio_ticks, audio_len = self.audio_ticks_visuzalize_idx(data)

        # create text labels
        text_ticks = self.text_ticks_visualize_idx(data)
        text_len = len(text_ticks)

        attention = batch_verb[attn_name][idx].cpu().numpy()
        attention = attention[:text_len, :audio_len]
        return self.attention_visualize_ax(
            ax, attention, audio_ticks, text_ticks, 'audio', 'text', title
        )

    def text_attention_visualize_idx(self, batch_verb, idx, ax, title, attn_name):
        # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#visualizing-attention
        dataset_idx = batch_verb['indices'][idx].item()
        data = self.dataset.get_verbose(dataset_idx)

        # create text labels
        text_ticks = self.text_ticks_visualize_idx(data)
        text_len = len(text_ticks)

        attention = batch_verb[attn_name][idx].cpu().numpy()
        attention = attention[:text_len, :text_len]
        return self.attention_visualize_ax(
            ax, attention, text_ticks, text_ticks, 'text', 'text', title
        )

    def attention_vis(self, batch_verb, attn_name):
        if attn_name not in batch_verb:
            return None

        worst_loss, _, worst_idx = self.worst_examples(1, batch_verb)[0]

        best_comp = lambda new, curr: new < curr
        best_loss, _, best_idx = self.worst_examples(1, batch_verb, comp_func=best_comp)[0]

        indices = [worst_idx, best_idx]
        names = [f'worst_{worst_loss}', f'best_{best_loss}']
        # one attention (24, 6)

        # fig = plt.figure(figsize=(24, 6))
        # ax = fig.add_subplot(111)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(22, 24))
        for ax, ind, name in zip(axes, indices, names):
            if attn_name == 'text_attn':
                img = self.text_attention_visualize_idx(batch_verb, ind, ax, name, attn_name)
            elif attn_name == 'cross_attn':
                img = self.cross_attention_visualize_idx(batch_verb, ind, ax, name, attn_name)
            else:
                raise ValueError(f'unknown attention type {attn_name}')

        # https://stackoverflow.com/a/13784887
        fig.colorbar(img, ax=axes.ravel().tolist())

        # fig.set_tight_layout(True)
        return fig

    def loss_vs_seq_len(self, batch_verbose):
        # x-axis loss

        # y-axis seq-len
        _, trgs = self.verbose_prep(batch_verbose, 'punc_preds', 'punc_trgs', indices)

        pass

# todo multiple attentions visualize
# todo only the last epoch, x_axis -> loss, y_axis -> sequence length
