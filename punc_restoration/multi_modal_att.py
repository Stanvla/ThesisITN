# %%
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel
try:
    from args import get_args
    from load_data import AudioDataPL
except ModuleNotFoundError as e:
    from punc_restoration.load_data import AudioDataPL
    from punc_restoration.args import get_args


class AttentionBlock(nn.Module):
    def __init__(self, roberta_cfg, w2v_cfg):
        super().__init__()
        self.roberta_cfg = roberta_cfg
        self.w2v_cfg = w2v_cfg

        self.num_attention_heads = roberta_cfg.num_attention_heads
        self.hidden_size = roberta_cfg.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.attn_do_prob = roberta_cfg.attention_probs_dropout_prob
        self.ff_do_prob = roberta_cfg.hidden_dropout_prob

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(roberta_cfg.hidden_dropout_prob)

        self.norm = nn.LayerNorm(
            self.hidden_size,
            eps=roberta_cfg.layer_norm_eps,
        )

    def forward(self, queries_src, keys_src, values_src, attn_mask, key_padding_mask=None):
        # decoder forward pytorch
        # https://github.com/pytorch/pytorch/blob/4bf90558e0cbafbf03fa7e4285367f12658bde54/torch/nn/modules/transformer.py#L352
        attn_output = self.self_attn(queries_src, keys_src, values_src, attn_mask=attn_mask)
        # attn_output.shape = [batch, seq_len, hidden]

        # dense according to
        # https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/bert/modeling_bert.py#L376
        # is inside nn.MultiHeadAttention

        # add norm with dropout
        return self.norm(self.dropout(attn_output) + queries_src)


class TransformerOutput(nn.Module):
    def __init__(self, do_prob, hidden_size, eps):
        super().__init__()
        self.do = nn.Dropout(do_prob)
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, inp_prev, inp_new):
        result = inp_prev + self.do(inp_new)
        result = self.norm(result)
        return result


class MMEncoderLayer(nn.Module):
    def __init__(self, roberta_cfg, vdim):
        super().__init__()

        # assert roberta_cfg.num_attention_heads == w2v_cfg.num_attention_heads
        # assert roberta_cfg.hidden_size == w2v_cfg.hidden_size
        # assert roberta_cfg.attention_probs_dropout_prob == w2v_cfg.attention_dropout
        # assert roberta_cfg.hidden_size % roberta_cfg.num_attention_heads == 0

        self.hidden_size = roberta_cfg.hidden_size
        self.intermediate_size = roberta_cfg.intermediate_size
        # self.intermediate_size = 2048
        self.attn_do_prob = roberta_cfg.attention_probs_dropout_prob
        self.ff_do_prob = roberta_cfg.hidden_dropout_prob
        self.hidden_do_prob = roberta_cfg.hidden_dropout_prob
        self.eps = roberta_cfg.layer_norm_eps

        self.num_attention_heads = roberta_cfg.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        # self attention block
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=self.attn_do_prob,
            batch_first=True,
        )
        self.self_attn_out = TransformerOutput(self.hidden_do_prob, self.hidden_size, self.eps)

        # cross attention block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=self.attn_do_prob,
            batch_first=True,
            kdim=vdim,
            vdim=vdim,
        )
        self.cross_attn_out = TransformerOutput(self.hidden_do_prob, self.hidden_size, self.eps)

        # feed forward block
        self.ff_proj_in = nn.Linear(self.hidden_size, self.intermediate_size)
        self.ff_fn = nn.GELU()
        self.ff_do = nn.Dropout(self.hidden_do_prob)
        self.ff_proj_out = nn.Linear(self.intermediate_size, self.hidden_size)

        # final output
        self.out = TransformerOutput(self.hidden_do_prob, self.hidden_size, self.eps)

    def forward(self, h_text, h_audio, audio_padding_mask=None, text_padding_mask=None, cross_mask=None, text_self_mask=None):

        # audio_padding_mask : shape = [batch_size, max_audio_len] ... 1 -> mask, 0 -> leave, 1s shows padding in the audio
        # (memory_key_padding_mask)

        # text_padding_mask : shape = [batch_size, max_text_len] ... 1 -> mask, 0 -> leave, 1s shows padding in the text
        # (tgt_key_padding_mask)

        # cross_mask : shape = [max_tex_len, max_audio_len] ... additive mask for multimodal cross attention
        #   -> matmul(text_queries, audio_keys.T) + cross_mask, if text token i can attend sound token j,
        #   values ~ -inf to cancel attention
        #   (memory_mask)

        # text_self_mask: shape = [max_text_len, max_text_len] ... additive mask for text self attention
        #   -> matmul(text_queries, text_keys.T) + self_text_mask, if text token i can attend text token j,
        #   values ~ -inf to cancel attention
        #   (tgt_mask)

        # pytorch forward implementation
        # https://github.com/pytorch/pytorch/blob/4bf90558e0cbafbf03fa7e4285367f12658bde54/torch/nn/modules/transformer.py#L352

        # tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # ...
        # tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        # ...
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # ...

        # masks explained
        # https://stackoverflow.com/a/68396781

        self_attn_output = self.self_attn(h_text, h_text, h_text, key_padding_mask=text_padding_mask, need_weights=False)[0]
        self_attn_output = self.self_attn_out(h_text, self_attn_output)
        cross_attn_output, attention_weights = self.cross_attn(self_attn_output, h_audio, h_audio, need_weights=True)
        cross_attn_output = self.cross_attn_out(self_attn_output, cross_attn_output)

        # ff
        projected = self.ff_proj_in(cross_attn_output)
        projected = self.ff_fn(projected)
        projected = self.ff_do(projected)
        projected = self.ff_proj_out(projected)

        # final_output
        new_hidden = self.out(cross_attn_output, projected)

        return new_hidden, attention_weights, h_audio, audio_padding_mask, text_padding_mask, cross_mask, text_self_mask


class MMEncoder(nn.Module):
    def __init__(self, args, roberta_conf):
        super().__init__()
        self.args = args
        vdim = args['audio_params']['cnn_model']['out_dims'][-1]
        self.layers = nn.ModuleDict({
            f'layer_{i}': MMEncoderLayer(roberta_conf, vdim) for i in range(args['mm_encoder']['n_layers'])
        })

    def forward(self, h_text, h_audio, audio_padding_mask=None, text_padding_mask=None, cross_mask=None, text_self_mask=None):
        attention_weights = None
        for layer_name, layer in self.layers.items():
            h_text, attention_weights, *_ = layer(h_text, h_audio, audio_padding_mask, text_padding_mask, cross_mask, text_self_mask)

        return h_text, attention_weights


# %%
if __name__ == '__main__':
    # %%
    params = get_args()
    audio_data = AudioDataPL(params)

    audio_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])
    roberta_model = AutoModel.from_pretrained(params['robeczech_path'])
    # %%
    test_loader = audio_data.test_dataloader()
    for i, batch in enumerate(test_loader):
        if i == 1:
            break
        # roberta inputs
        txt_inps, punc_trgs, cap_trgs, txt_masks, txt_seq_lens = batch['text_ids'], batch['punc_trgs'], batch['cap_trgs'], batch['masks'], batch['seq_lens']
        # w2v2 inputs
        mp3_inps, mp3_masks = batch['audio'], batch['audio_mask']

    # %%
    hidden_text = roberta_model(txt_inps, txt_masks)
    hidden_audio = audio_model(mp3_inps)
    new_txt_masks = txt_masks == 0
    # %%

    mm_encoder = MMEncoder(params, roberta_model.config, audio_model.config)

    x = mm_encoder.forward(
        hidden_text.last_hidden_state,
        hidden_audio.last_hidden_state,
        text_padding_mask=new_txt_masks,
    )

    # %%
    for m in mm_encoder.layers:
        print(m)
    # %%
    # todo:
    #  1. get timestamps for each word
    #  1. compute "timestamps" in the hidden space
    #  1. pass input through w2v2
    #  1. extract fix-sized windows for each word
    #  1. ... multimodal attention
    # %%


    # %%
    # %%
    # %%
    # %%
    pass