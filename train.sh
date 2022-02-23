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
        --dataflag=grayscale \
        --amp=O1 
        # --model=convnext
        # --resume=results/RCF20220123_1024-bs-8-lr-0.002-iter_size-1-opt-adamw/checkpoint_epoch220.pth