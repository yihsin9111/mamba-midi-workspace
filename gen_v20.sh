# loss = (3)
# python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay.py --device cuda:0 --project_name v20 --ckpt 52
# python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay_test.py --device cuda:1 --project_name v20 --ckpt 52

# loss = (3.5)
# python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay.py --device cuda:1 --project_name v20 --ckpt 28

# loss = (4)
# python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay.py --device cuda:1 --project_name v20 --ckpt 18

# loss = (4.5)
python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay.py --device cuda:1 --project_name v20 --ckpt 14


python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay_test.py --device cuda:1 --project_name v20 --ckpt 28
python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay_test.py --device cuda:1 --project_name v20 --ckpt 18
python /mnt/gestalt/home/lonian/mamba/model/generate_simba_delay_test.py --device cuda:1 --project_name v20 --ckpt 14