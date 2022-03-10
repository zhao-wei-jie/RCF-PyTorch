CUDA_VISIBLE_DEVICES=2 python train.py \
        --batch-size=4\
        --opt=adamw \
        --max-epoch=1000\
        --gpu=0\
        --print-freq=500\
        --lr=2e-4 \
        --stepsize=50 \
        --gamma=0.9 \
        --weight-decay=1e-2 \
        --iter-size=1\
        --amp=O1 \
        --model=hrnet_ocr \
        --fuse_num=5 \
        --short_cat=2 \
        --dataflag=color \
        --scale=True\
        # --augs=flip\
        # --resume=results/UNET20220308_2121-bs-4-lr-0.0002-dataflag-grayscale-aug-False/checkpoint_epoch70.pth
        # --aug=True
        
        
        
        # --pretrain=results/RCF20220223_2140-bs-4-lr-0.002-iter_size-1-opt-adamw/checkpoint_epoch594.pth