python train.py --cuda \
                -d imagenet \
                --root /home/k545/datasets/imagenet/ \
                --optimizer adamw \
                --learning_rate 0.001 \
                --max_epoch 300 \
                --batch_size 256 \
                --lr_schedule cos \
                --img_size 224 \
                --num_patch 16 \
                --dim 384 \
                --depth 4 \
                --heads 8 \
                --dim_head 64 \
                --channels 3 \
                --mlp_dim 384 \
                --pool cls 
