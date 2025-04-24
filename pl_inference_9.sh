
# path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_9_incontext/lightning_logs/version_4/checkpoints/epoch=198-step=85000.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# CUDA_VISIBLE_DEVICES=2 python pl_inference.py ${path_prefix_simba} baseline prefix_simba_85000

# CUDA_VISIBLE_DEVICES=2 python pl_inference_musiccaps_30s.py ${path_prefix_simba} baseline prefix_simba_85000

path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/version_4/checkpoints/epoch=198-step=85000.ckpt"
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_transformer="/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/version_6/checkpoints/epoch=198-step=85000.ckpt"
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi

CUDA_VISIBLE_DEVICES=2 python pl_inference_musiccaps_30s.py ${path_prefix_simba} 25s_85000 prefix_simba
CUDA_VISIBLE_DEVICES=2 python pl_inference_musiccaps_30s.py ${path_cross_transformer} 25s_85000 cross_transformer


path_prefix_simba="//mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/version_3/checkpoints/epoch=46-step=20000.ckpt"
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_transformer="/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/version_4/checkpoints/epoch=46-step=20000.ckpt"
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi

CUDA_VISIBLE_DEVICES=2 python pl_inference_musiccaps_30s.py ${path_prefix_simba} 25s_20000 prefix_simba
CUDA_VISIBLE_DEVICES=2 python pl_inference_musiccaps_30s.py ${path_cross_transformer} 25s_20000 cross_transformer