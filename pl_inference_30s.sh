folder="exp1_85000steps_prompt"
epoch_num="*"
steps_num="85000"
cuda_device=2


# path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_mamba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
# path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi

# incontext mamba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# incontext simba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} prefix_simba
# cross_attn simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# cross_attn transformer
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_cross_transformer} ${folder} cross_transformer






folder="exp1_20000steps"
epoch_num="*"
steps_num="20000"
cuda_device=2


# path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_mamba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
# path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi

# incontext mamba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# incontext simba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} prefix_simba
# cross_attn simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# cross_attn transformer
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_cross_transformer} ${folder} cross_transformer