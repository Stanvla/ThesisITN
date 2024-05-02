# %%
"""
Perform the recognition and time extraction using the `wav2vec2_inference.py` script
"""
import os
import subprocess
import argparse
import shutil
import logging
from dataclasses import dataclass


parser = argparse.ArgumentParser()
parser.add_argument("--jobs_cnt", default=300)
parser.add_argument("--output_base_dir", default='/lnet/express/work/people/stankov/alignment/new-parczech/baseline')

@dataclass
class Params:
    gold_vert_path = '/lnet/express/work/people/stankov/alignment/results/full/merged/'
    mp3_path = '/lnet/work/people/kopp/ParCzech/psp.cz/mp3/original_files/www.psp.cz/eknih'
    base_script = '/lnet/express/work/people/stankov/alignment/Thesis/wav2vec2_ctc/wav2vec_inference.py'


slurm_config = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -D {log_dir}
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

source /lnet/express/work/people/stankov/python-venvs/new-gpu-cache/bin/activate
"""


if __name__ == '__main__':
    # %%
    args = parser.parse_args([] if "__file__" not in globals() else None)
    params = Params()
    log_dir_slurm = os.path.join(args.output_base_dir, 'slurm_logs')
    scripts_dir = os.path.join(args.output_base_dir, 'create_dataset_scripts')

    # --mp3_source
    # --output_dir

    os.makedirs(args.output_base_dir, exist_ok=True)
    os.makedirs(log_dir_slurm, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    verticals = {}
    for file in sorted(os.listdir(params.gold_vert_path)):
        if not file.endswith('.vert'):
            continue
        verticals[file.replace('.vert', '')] = os.path.join(params.gold_vert_path, file)

    # read all source mp3 files
    # mp3_files[mp3_name] = path to mp3 file
    mp3_files = {}
    for root, subdir, files in os.walk(params.mp3_path):
        for f in files:
            if f.endswith('.mp3') and f.replace('.mp3', '') in verticals:
                mp3_files[f.replace('.mp3', '')] = os.path.join(root, f)


    verticals_lists = [[] for i in range(args.jobs_cnt)]
    for i, vert in enumerate(verticals.keys()):
        verticals_lists[i % args.jobs_cnt].append(vert)


    scripts = []
    for i, lst in enumerate(verticals_lists):
        commands = [f"python {params.base_script} --mp3_source={mp3_files[vert]} --output_dir={args.output_base_dir} --job_idx={i}" for vert in lst]
        commands_echo_start = [f"echo {mp3_files[vert]}" for i, vert in enumerate(lst)]
        commands_echo_end = [f"echo {vert} finished, {i+1:3}/{len(lst)}, {100 * (i+1)/len(lst):.2f}%" for i, vert in enumerate(lst)]
        commands = [f'{echo_s}\n{cmd}\n{echo_e}' for cmd, echo_s, echo_e in zip(commands, commands_echo_start, commands_echo_end)]
        script = slurm_config.format(
            job_name=f'parcz-{i}',
            log_dir=log_dir_slurm,
        )
        script += '\n' + '\n'.join(commands)
        scripts.append(script)
        # print(script)
        # print('---'*10)

        script_name = f'{i:>03}.sh'
        script_path = os.path.join(scripts_dir, script_name)
        with open(script_path, 'w') as f:
            f.write(script)

        subprocess.run(f'sbatch {script_path}'.split())
