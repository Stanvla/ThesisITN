# %%
import torch
from transformers import Wav2Vec2Model
from tqdm import tqdm
import os
import pandas as pd
import math
import pytorch_lightning as pl
import logging
try:
    from args import get_args
    from load_data import AudioDataPL
except ModuleNotFoundError as e:
    from punc_restoration.load_data import AudioDataPL
    from punc_restoration.args import get_args


def compute_hidden_frames(wave_len, feature_extractor):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    layers = feature_extractor.conv_layers
    for layer in layers:
        conv = layer.conv
        dilation, kernel_size, stride, padding = conv.dilation[0], conv.kernel_size[0], conv.stride[0], conv.padding[0]
        wave_len = math.floor(
            (wave_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
    return wave_len


class SimpleW2VInf(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(args['audio_params']['model_weights_path'])

    def forward(self, inputs):
        return self.model(inputs).last_hidden_state

    def test_step(self, batch, idx, **kwargs):
        inputs = batch['audio']
        hidden_lens = [compute_hidden_frames(sl, self.model.feature_extractor) for sl in batch['audio_sl']]

        paths = [p.replace('mp3', 'parquet') for p in batch['audio_path']]

        # leave only audios that are missing
        # missing_mask = [not os.path.isfile(p) for p in paths]
        #
        # paths = [p for p, missing in zip(paths, missing_mask) if missing]
        # hidden_lens = [hl for hl, missing in zip(hidden_lens, missing_mask) if missing]
        # inputs = inputs[missing_mask]

        if paths != []:
            hiddens = self(inputs)

            # columns = [f'f{i}' for i in range(768)]
            # for feature, p, hl in zip(hiddens, paths, hidden_lens):
            #     df = pd.DataFrame(data=feature[:hl].cpu().numpy(), columns=columns)
            #     df.to_parquet(p)
                # torch.save(feature[:hl], p)

        if self.global_rank == 0 and idx % 20 == 0:
            logging.debug(f'batch {idx:4}')


# %%
if __name__ == '__main__':
    # %%
    params = get_args()

    logging.basicConfig(
        filename=params['logging']['text_logger'].replace('debug_logger.log', 'audio-features.log'),
        filemode='a', level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%H:%M %d.%m'
    )


    audio_data = AudioDataPL(params)
    # %%
    pl_model = SimpleW2VInf(params)

    trainer_params = dict(
        num_sanity_val_steps=0,
        deterministic=True,
        devices=None,
        precision=16,
        accelerator='cpu',
        # enable_progress_bar=False,
    )
    if torch.cuda.is_available():
        trainer_params['accelerator'] = 'gpu'
        trainer_params['devices'] = torch.cuda.device_count()
        if trainer_params['devices'] > 1:
            trainer_params['strategy'] = 'ddp'

    trainer = pl.Trainer(**trainer_params)

    trainer.test(pl_model, audio_data, verbose=False)
    # %%


    # loaders = {
    #     'Test': audio_data.test_dataloader(),
    # }
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = Wav2Vec2Model.from_pretrained(params['audio_params']['model_weights_path']).to(device)
    # # %%
    # for name, loader in loaders.items():
    #     # for i, batch in enumerate(tqdm(loader, desc=name)):
    #     for i, batch in enumerate(loader):
    #         if i == 4:
    #             break
    #
    #         inputs = batch['audio']
    #         hidden_lens = [compute_hidden_frames(sl, model.feature_extractor) for sl in batch['audio_sl']]
    #
    #         paths = [p.replace('mp3', 'pt') for p in batch['audio_path']]
    #
    #         # leave only audios that are missing
    #         missing_mask = [not os.path.isfile(p) for p in paths]
    #
    #         paths = [p for p, missing in zip(paths, missing_mask) if missing]
    #         hidden_lens = [hl for hl, missing in zip(hidden_lens, missing_mask) if missing]
    #         inputs = inputs[missing_mask]
    #
    #         with torch.no_grad():
    #             hidden = model(inputs).last_hidden_state
    #
    #         for feature, p, hl in zip(hidden, paths, hidden_lens):
    #             print(feature.shape)
    #             print(hl)
    #             print(p)
    #         print()

    # %%
    # %%
    # %%
    # %%
    # %%
