#!/bin/bash
#$ -N LSPE-gape-b128-1
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-3:1

pos_enc_dim=(0 6 10 12)
fname=$(pwd)/b128-bnorm-alt-edge_${SGE_TASK_ID}_${pos_enc_dim[${SGE_TASK_ID}]}_DEBUG.log
touch $fname
fsync -d 10 $fname &

conda activate gnn
cd /afs/crc.nd.edu/user/p/psoga/gnn-lspe

python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_LSPE_withLapEigLoss.json --pe_init gape --n_gape 1 --pos_enc_dim ${pos_enc_dim[${SGE_TASK_ID}]}
