#!/bin/bash
#$ -N PNA_ZINC_lspe-ngape1_3
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/3_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/PNA_ZINC_lspe-ngape1.json --job_num 3 --pos_enc_dim 3 --log_file $fname
