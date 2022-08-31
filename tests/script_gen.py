import os
import argparse

def get_script_text(job_name, v, command):
    text = f"""#!/bin/bash
#$ -N {job_name}_{v}
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/{v}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/benchmarking-gnns

{command}
"""
    return text

def main(args):
    file = args.file
    dir = os.path.dirname(file)

    # parse file into text format
    lines = open(file, 'r').readlines()

    job_name = lines[1].split(' ')[-1].strip()
    values = [int(x) for x in lines[6].split('=')[1].strip('()\n').split(' ')][1:]

    command = lines[14].strip().split(' ')

    job_num_idx = [i for i, x in enumerate(command) if 'job_num' in x][0] + 1
    pos_enc_dim_idx = [i for i, x in enumerate(command) if 'pos_enc_dim' in x][0] + 1

    for value in values:
        command[job_num_idx] = f'{value}'
        command[pos_enc_dim_idx] = f'{value}'

        with open(f'{dir}/{job_name}_{value}.sh', 'w') as f:
            f.write(get_script_text(job_name, value, ' '.join(command)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to script to split")
    main(parser.parse_args())