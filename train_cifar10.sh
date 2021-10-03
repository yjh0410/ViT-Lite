python train.py --cuda \
                -d cifar10 \
                --root /home/k545/datasets/cifar10/ \
                --optimizer adamw \
                --learning_rate 0.001 \
                --max_epoch 90 \
                --batch_size 256 \
                --lr_schedule cos \
                --img_size 32 \
                --num_patch 4 \
                --dim 256 \
                --depth 4 \
                --heads 4 \
                --dim_head 32 \
                --channels 3 \
                --mlp_dim 256 \
                --pool cls 