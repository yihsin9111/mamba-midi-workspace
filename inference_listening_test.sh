epoch_num="*"
cuda_device=2

steps_num="85000"
path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
# path_9_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_9_incontext/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi


CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_listening_test_30s.py ${path_prefix_simba} "demo_page" 2_stages
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_listening_test_30s.py ${path_9_prefix_simba} "0314" 9_levels