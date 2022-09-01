#!/bin/bash
#$ -N GatedGCN_ZINC_lspe-eigloss-ngape1-eq_156
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/156_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_lspe-eigloss-ngape1-eq.json --job_num 156 --pos_enc_dim 156 --log_file $fname
