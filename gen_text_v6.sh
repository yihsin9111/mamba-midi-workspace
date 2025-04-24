python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 30 --num 10 --batch 2 --g_scale 1.5
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A_g15
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B_g15

python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 30 --num 10 --batch 2 --g_scale 1.0
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A_g10
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B_g10

python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 30 --num 10 --batch 2 --g_scale 0
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_A_g0
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/30_B_g0


python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 26 --num 10 --batch 2 --g_scale 1.5
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A_g15
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B_g15

python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 26 --num 10 --batch 2 --g_scale 1.0
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A_g10
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B_g10

python /mnt/gestalt/home/lonian/mamba/model/generate_conditional_delay.py --device cuda:0 --project_name text_v6 --ckpt 26 --num 10 --batch 2 --g_scale 0
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_A_g00
mv /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B /mnt/gestalt/home/lonian/mamba/model/text_v6_results/26_B_g00