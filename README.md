
### Official Repository for My Master's Thesis Code

This repository contains the source code used in my master's thesis.

#### Folder Structure

- **wav2vec_ctc**: This folder is dedicated to creating the special ITN dataset from ParCzech 3.0.
  - `wav2vec_ctc/wav2vec_inference.py`: Runs inference using a trained wav2vec 2.0 model to produce recognized transcripts and alignment.
  - `wav2vec_ctc/dataset-review.py`: Filters and modifies aligned segments. Finally, generates training, validation, and testing sets from clean segments.

- **punc_restoration**: Contains scripts for training and evaluating the ITN models.
  - `punc_restoration/sequence_labeling.py`: Includes code for training and running inference on all models, both text-only and multimodal.
  - `punc_restoration/multi_modal_attn.py`: Implements the multimodal cross-attention module.
  - `punc_restoration/load_data.py`: Defines the structure for multimodal and text-only datasets.

