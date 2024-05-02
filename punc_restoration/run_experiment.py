import os
import argparse
import subprocess
import time
from subprocess import PIPE, run
from pprint import pprint
try:
    from args import get_args, short_name_gen
except ModuleNotFoundError as e:
    from punc_restoration.args import get_args, short_name_gen

parser = argparse.ArgumentParser()

# .......... cluster params ..........
parser.add_argument('--min_gram', default=16, type=int)
parser.add_argument('--max_gram', default=24, type=int)
parser.add_argument('--ngpus', default=2, type=int)
parser.add_argument('--cpus', default=8, type=int)
# .......... script params ..........
parser.add_argument('--ram', default=80, type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--mode", choices=['txt', 'sound'], type=str, required=True)
parser.add_argument("--bs", type=int, required=True)


SLURM_CONFIG = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -D {log_dir}
#SBATCH -o {slurm_log_file}.out
#SBATCH -e {slurm_log_file}.err
#SBATCH -p gpu-ms
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={ram}G
#SBATCH --gres=gpu:{ngpus}
#SBATCH --constraint="{gpu_ram}"


echo $HOSTNAME
echo

nvidia-smi
echo

# .......... create backup ..........
from_dir={from_dir}
backup_dir={backup_dir}

for file in $from_dir/*.py; do cp "$file" "$backup_dir/$(basename $file)"; done
for file in $from_dir/*.yaml; do cp "$file" "$backup_dir/$(basename $file)"; done
# ...................................

echo $(ls $backup_dir)
echo 

source /lnet/express/work/people/stankov/python-venvs/2023-python3.8.14/bin/activate
# echo "python $from_dir/sequence_labeling.py --name {cmd_name} --mode {cmd_mode} --bs {cmd_bs} --enable-pb False --text-logger {txt_logger}"
python $from_dir/sequence_labeling.py --name={cmd_name} --mode={cmd_mode} --bs={cmd_bs} --enable_pb=False --text_logger="{txt_logger}"
"""

# todo
#  1. log cfg params
#  2. log code


def generate_gram_range(min_gram, max_gram):
    if min_gram == max_gram:
        return f"gpuram{min_gram}G"

    available_grams = [16, 24, 40, 48]
    start_index = available_grams.index(min_gram)
    end_index = available_grams.index(max_gram)
    all_grams = '|'.join([f'gpuram{gram}G' for gram in available_grams[start_index: min(end_index + 1, len(available_grams))]])
    return all_grams


def countdown(time_sec):
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        if time_sec % 5 == 0:
            print(timeformat)
        time.sleep(1)
        time_sec -= 1

    print("run job")


if __name__ == '__main__':

    cmd_args = parser.parse_args([] if "__file__" not in globals() else None)
    yaml_args = get_args()

    pprint(cmd_args)
    pprint(yaml_args)
    input(':::::::::::::::::::::::::::::: Do you agree? ::::::::::::::::::::::::::::::')

    experiment_name, cur_time = short_name_gen(yaml_args, cmd_args, ngpus=cmd_args.ngpus)
    experiment_name = f'{experiment_name}__{cur_time}'
    # .................... constants ....................
    BASE_DIR = '/lnet/express/work/people/stankov/alignment/Thesis/punc_restoration'
    SLURM_LOG_DIR = os.path.join(BASE_DIR, 'slurm_logs', experiment_name)
    SLURM_SCRIPT_PATH = os.path.join(SLURM_LOG_DIR, 'slurm_script.bash')
    BACKUP_DIR = os.path.join(SLURM_LOG_DIR, 'backup')
    # ....................................................

    os.makedirs(BACKUP_DIR, exist_ok=True)

    script = SLURM_CONFIG.format(
        name=cmd_args.name,
        log_dir=SLURM_LOG_DIR,
        slurm_log_file=os.path.join(SLURM_LOG_DIR, 'slurm-logs'),
        cpus=cmd_args.cpus,
        ram=cmd_args.ram,
        ngpus=cmd_args.ngpus,
        gpu_ram=generate_gram_range(cmd_args.min_gram, cmd_args.max_gram),
        slurm_script=SLURM_SCRIPT_PATH,
        from_dir=BASE_DIR,
        backup_dir=BACKUP_DIR,
        cmd_name=cmd_args.name,
        cmd_mode=cmd_args.mode,
        cmd_bs=cmd_args.bs,
        txt_logger=os.path.join(SLURM_LOG_DIR, 'torch-logs.txt'),
    )

    with open(SLURM_SCRIPT_PATH, 'w') as f:
        f.write(script)

    subprocess.run(f'sbatch {SLURM_SCRIPT_PATH}'.split())