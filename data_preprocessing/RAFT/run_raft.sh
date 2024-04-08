NAME=heygen
ROOT_DIR=/root/CoDeF/all_sequences
CODE_DIR=/root/CoDeF/data_preprocessing/RAFT

IMG_DIR=$ROOT_DIR/${NAME}/${NAME}
FLOW_DIR=$ROOT_DIR/${NAME}/${NAME}_flow
CONF_DIR=${FLOW_DIR}_confidence

CUDA_VISIBLE_DEVICES=0 \
python ${CODE_DIR}/demo.py \
--model=${CODE_DIR}/models/raft-sintel.pth \
--path=$IMG_DIR \
--outdir=$FLOW_DIR \
--name=$NAME \
--confidence \
--outdir_conf=$CONF_DIR
