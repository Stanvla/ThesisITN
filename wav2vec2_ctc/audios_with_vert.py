# %%
import os

class Mp3:
    def __init__(self, mp3_path):
        self.path = mp3_path

    @property
    def name(self):
        return os.path.basename(self.path).replace('.mp3', '')

    @property
    def parliament(self):
        return self.path.split('/')[11]

    def __repr__(self):
        return self.path

# %%
all_verticals = [f.replace('.vert', '') for f in os.listdir("/lnet/express/work/people/stankov/alignment/results/full/merged") if f.endswith('.vert')]
all_verticals_set = set(all_verticals)


all_mp3s = []
mp3_source_dir = '/lnet/work/people/kopp/ParCzech/psp.cz/mp3/original_files/www.psp.cz/eknih'
for i, (root, dirs, files) in enumerate(os.walk(mp3_source_dir)):
    for f in files:
        if f.endswith('.mp3'):
            all_mp3s.append(Mp3(os.path.join(root, f)))

# all_mp3s_set = set(all_mp3s)
# %%
parliaments = {}
for mp3 in all_mp3s:
    if mp3.parliament not in parliaments:
        parliaments[mp3.parliament] = [mp3]
    else:
        parliaments[mp3.parliament] += [mp3]
# mp3s_without_vert = all_mp3s_set - all_verticals_set
# %%
# %%