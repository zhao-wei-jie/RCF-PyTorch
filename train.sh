python train.py \
        --batch-size=4\
        --opt=adamw \
        --max-epoch=1000\
        --gpu=0\
        --print-freq=500\
        --lr=2e-3 \
        --stepsize=50 \
        --gamma=0.9 \
        --weight-decay=1e-2 \
        --iter-size=1\
        --amp=O1 \
        --model=rcf \
        # --dataflag=grayscale \
        # --resume=results/RCF20220223_1214-bs-4-lr-0.002-iter_size-1-opt-adamw/checkpoint_epoch253.pth
        # --pretrain=results/RCF20220223_2140-bs-4-lr-0.002-iter_size-1-opt-adamw/checkpoint_epoch594.pth