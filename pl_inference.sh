# folder="exp1_85000steps"
# epoch_num="*"
# steps_num="85000"
# cuda_device=2


# # path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_prefix_mamba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# # path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_cross_simba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# # incontext mamba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # incontext simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# # cross_attn simba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# # cross_attn transformer
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer






# folder="exp1_60000steps"
# epoch_num="*"
# steps_num="60000"
# # cuda_device=0


# # path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_prefix_mamba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# # path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_cross_simba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# # incontext mamba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # incontext simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# # cross_attn simba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# # cross_attn transformer
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer



# folder="exp1_70000steps"
# epoch_num="*"
# steps_num="70000"
# # cuda_device=0


# # path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_prefix_mamba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# # path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_cross_simba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# # incontext mamba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # incontext simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# # cross_attn simba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# # cross_attn transformer
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer


# folder="exp1_80000steps"
# epoch_num="*"
# steps_num="80000"
# cuda_device=2


# # path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_prefix_mamba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# # path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_cross_simba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# # incontext mamba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # incontext simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# # cross_attn simba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# # cross_attn transformer
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer

# folder="exp1_50000steps"
# epoch_num="*"
# steps_num="50000"
# cuda_device=2


# # path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_prefix_mamba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# # path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# # if [ -f ${path_cross_simba} ]; then
# #     echo "Yes"
# # else
# #     echo "No"
# # fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# # incontext mamba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # incontext simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# # cross_attn simba
# # CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba
# # cross_attn transformer
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer


# folder="exp1_40000steps"
# epoch_num="*"
# steps_num="40000"
# cuda_device=2

# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer

# folder="exp1_30000steps"
# epoch_num="*"
# steps_num="30000"
# cuda_device=2

# path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_prefix_simba} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi
# path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
# if [ -f ${path_cross_transformer} ]; then
#     echo "Yes"
# else
#     echo "No"
# fi

# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer

steps_num="10000"
folder="exp1_10000steps"
epoch_num="*"
cuda_device="2"

path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_mamba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi


# incontext simba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# cross_attn transformer
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer
# incontext mamba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# # cross_attn simba
# CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba

steps_num="5000"
folder="exp1_5000steps"
cuda_device="2"

path_prefix_mamba=/mnt/gestalt/home/lonian/mamba/model/ckpts/mamba_1_incontext_smalldata_bf16_1129/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_mamba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_prefix_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_incontext_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_prefix_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_simba=/mnt/gestalt/home/lonian/mamba/model/ckpts/simba_1_cross_smalldata_bf16/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_simba} ]; then
    echo "Yes"
else
    echo "No"
fi
path_cross_transformer=/mnt/gestalt/home/lonian/mamba/model/ckpts/transformer_1_cross_0121/lightning_logs/*/checkpoints/epoch=${epoch_num}-step=${steps_num}.ckpt
if [ -f ${path_cross_transformer} ]; then
    echo "Yes"
else
    echo "No"
fi


# incontext simba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_simba} ${folder} prefix_simba
# cross_attn transformer
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_transformer} ${folder} cross_transformer
# incontext mamba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_prefix_mamba} ${folder} prefix_mamba
# cross_attn simba
CUDA_VISIBLE_DEVICES=${cuda_device} python pl_inference.py ${path_cross_simba} ${folder} cross_simba