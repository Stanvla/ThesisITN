# %%
import os
import pickle
from tqdm import tqdm
# %%
with open('outputs/text.pkl', 'rb') as f:
    loaded = pickle.load(f)
loaded[0]

# %%
dataset_dir = '/lnet/express/work/people/stankov/alignment/results/full/RELEASE-LAYOUT/'
text_data = []
for folder in sorted(os.listdir(dataset_dir)):
    folder_path = os.path.join(dataset_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for mp3_folder in tqdm(os.listdir(folder_path), desc=folder):
        mp3_folder_path = os.path.join(folder_path, mp3_folder)
        if not os.path.isdir(mp3_folder_path):
            continue
        for segment_folder in sorted(os.listdir(mp3_folder_path)):
            segment_folder_path = os.path.join(mp3_folder_path, segment_folder)
            if not os.path.isdir(segment_folder_path):
                continue
            for file in sorted(os.listdir(segment_folder_path)):
                if file.endswith('.prt'):
                    file_path = os.path.join(segment_folder_path, file)
                    with open(file_path, 'r') as f:
                        text_data.append(f.read())

pickled_text_file = 'outputs/text.pkl'
with open(pickled_text_file, 'wb') as f:
    pickle.dump(text_data, f, protocol=pickle.HIGHEST_PROTOCOL)

