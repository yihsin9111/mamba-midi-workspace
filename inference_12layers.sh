steps_num="85000"
folder="exp4_${steps_num}steps"
epoch_num="*"
cuda_device=0

path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/prefix_simba_1_12layer/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 12_layers


# steps_num="80000"
# folder="exp4_${steps_num}steps"
# path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/prefix_simba_1_12layer/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 12_layers


# steps_num="60000"
# folder="exp4_${steps_num}steps"
# path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/prefix_simba_1_12layer/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 12_layers

# steps_num="40000"
# folder="exp4_${steps_num}steps"
# path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/prefix_simba_1_12layer/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 12_layers


# steps_num="20000"
# folder="exp4_${steps_num}steps"
# path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/prefix_simba_1_12layer/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 12_layers

