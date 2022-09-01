#!/bin/bash
#$ -N GatedGCN_ZINC_gatedgcn_lspe_gape_512
#$ -q gpu
#$ -l gpu_card=1

fname=$(pwd)/512_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python3 main_ZINC_graph_regression.py --config tests/test-configs/GatedGCN_ZINC_gatedgcn_lspe_gape.json --job_num 512 --pos_enc_dim 512 --log_file $fname
