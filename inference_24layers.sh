epoch_num="*"
cuda_device=3


for steps_num in 20000 40000 60000 80000 85000
do
    # steps_num="85000"
    folder="exp4_${steps_num}steps"
    path_prefix_simba="/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt"
    if [ -f ${path_prefix_simba} ]; then
        echo "Yes"
    else
        echo "No"
    fi
    CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference_musiccaps_30s.py ${path_prefix_simba} ${folder} 24_layers
done