import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel
from torchaudio import transforms as T
try:
    from args import get_args
    from load_data import AudioDataPL
except ModuleNotFoundError as e:
    from punc_restoration.load_data import AudioDataPL
    from punc_restoration.args import get_args


class AudioBlock(nn.Module):
    def __init__(self, args, idx):
        super().__init__()

        self.conv = nn.Conv1d(
            args['in_dims'][idx],
            args['out_dims'][idx],
            kernel_size=args['kernels'][idx],
            stride=args['strides'][idx],
            bias=args['bias']
        )

        self.activation = nn.GELU()

        self.norm = None
        self.norm_type = args['normalization']

        if args['use_norm']:
            if self.norm_type == 'LayerNorm':
                self.norm = nn.LayerNorm(
                    args['out_dims'][idx]
                )
            elif self.norm_type == 'BatchNorm':
                self.norm = nn.BatchNorm1d(
                    args['out_dims'][idx]
                )
            else:
                raise ValueError(f'Normalization `{args["norm_type"]}` is not supported.')

        self.do = None
        if args['use_do'][idx]:
            self.do = nn.Dropout(args['dropout'])

    def forward(self, inputs):
        hiddens = self.conv(inputs)

        if self.norm is not None:
            if self.norm_type == 'LayerNorm':
                hiddens = hiddens.transpose(-2, -1)
                hiddens = self.norm(hiddens)
                hiddens = hiddens.transpose(-2, -1)
            else:
                hiddens = self.norm(hiddens)

        hiddens = self.activation(hiddens)

        if self.do is not None:
            hiddens = self.do(hiddens)

        return hiddens


class AudioEncoder(nn.Module):
    def __init__(self, args, src_model):
        super().__init__()
        self.params = args['audio_params']
        self.backbone_type = self.params['backbone']

        if self.backbone_type == 'melspectrogram':
            params = {**self.params['mel_args'], **self.params['spec_args']}
            self.backbone = T.MelSpectrogram(**params)
            # set input dimensions based on nmels
            self.params['cnn_model']['in_dims'][0] = self.params['mel_args']['n_mels']
        elif self.backbone_type == 'w2v2':
            self.backbone = src_model.feature_extractor
        else:
            raise ValueError(f'Unknown backbone type :: `{self.backbone_type}`')

        conv_layers = []
        self.block_params = self.params['cnn_model']

        for i in range(self.block_params['layers']):
            conv_layers.append(AudioBlock(self.block_params, i))

        self.conv_layers = nn.ModuleList(conv_layers)

        self.trans_params = self.params['t_head']
        self.transformer = None
        self.transformer_do = None
        self.temb = None
        if self.trans_params['use_transformer']:
            dimension = self.block_params['out_dims'][self.block_params['layers'] - 1]
            layer = nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=self.trans_params['nhead'],
                dim_feedforward=self.trans_params['dim_feedforward'],
                activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer=layer,
                num_layers=self.trans_params['layers']
            )

            self.transformer_do = nn.Dropout(args['audio_params']['emb_do'])

            if self.params['t_emb']['use_temb']:
                self.temb = nn.Embedding(
                    num_embeddings=self.params['t_emb']['num_embeddings'],
                    embedding_dim=self.block_params['out_dims'][-1],
                )
                self.tembdo = nn.Dropout(
                    p=self.params['t_emb']['do']
                )

    def freeze_backbone(self):
        if self.backbone_type == 'w2v2' and self.block_params.freeze_backbone:
            self.backbone._freeze_parameters()

    def forward(self, inputs, pos_emb=None, raw_len=None, f_len=None, device=None):
        # print(inputs.shape)
        if self.backbone_type == 'w2v2' and self.block_params.freeze_backbone:
            with torch.no_grad():
                hiddens = self.backbone(inputs)
        else:
            hiddens = self.backbone(inputs)

        for conv_layer in self.conv_layers:
            hiddens = conv_layer(hiddens)

        if self.transformer is not None:
            hiddens = hiddens.transpose(-2, -1)

            if self.temb is not None and f_len is not None:
                max_len = hiddens.shape[1]
                indices = torch.tensor([
                    list(range(l)) + [self.params['t_emb']['num_embeddings'] - 1]*(max_len - l)
                    for l in f_len
                ]).to(device)
                hiddens = self.tembdo(hiddens + self.temb(indices))

            if pos_emb is not None:
                hiddens = self.transformer_do(hiddens + pos_emb)

            return self.transformer(hiddens)

        return hiddens.transpose(-2, -1)



# %%
if __name__ == '__main__':
    # %%
    params = get_args()

    audio_model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path'])
    audio_model.mask_time_prob = 0
    audio_encoder = AudioEncoder(params, audio_model)
    # %%
    inputs = torch.rand(2, 16_000)
    # %%
    output = audio_encoder(inputs)
    # %%
    # %%
