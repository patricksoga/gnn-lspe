#!/bin/bash
#$ -N GatedGCN_ZINC_gatedgcn_lspe_gapen3_20
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/20_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_gatedgcn_lspe_gapen3.json --job_num 20 --pos_enc_dim 20 --log_file $fname
