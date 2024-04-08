GPUS=0

NAME=heygen
EXP_NAME=base

ROOT_DIRECTORY="all_sequences/$NAME/$NAME"
LOG_SAVE_PATH="logs/test_all_sequences/$NAME"

# WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/${NAME}.ckpt
WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/step=200000.ckpt

python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --weight_path $WEIGHT_PATH \
                --gpus $GPUS \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME} \
                --save_deform False
