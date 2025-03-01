#!/bin/bash
#$ -N GatedGCN_ZINC_lspe-eigloss-ngape1-rwk_6
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/6_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_lspe-eigloss-ngape1-rwk.json --job_num 6 --pos_enc_dim 6 --log_file $fname
