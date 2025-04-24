# mkdir /mnt/gestalt/home/lonian/mamba/model/v19_results/14_A/all_token
# mkdir /mnt/gestalt/home/lonian/mamba/model/v19_results/14_A/all_audio
# mkdir /mnt/gestalt/home/lonian/mamba/model/v19_results/14_A/token
# mkdir /mnt/gestalt/home/lonian/mamba/model/v19_results/14_A/audio
NAME=text_v9

for EPOCH in 100 200
do
    for Dset in A B
    do
        # mkdir /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/all_token
        # mkdir /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/all_audio
        mkdir /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/token
        mkdir /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/audio
        # mv /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/*_all.npy /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/all_token
        # mv /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/*_all.wav /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/all_audio
        mv /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/*.npy /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/token
        mv /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/*.wav /mnt/gestalt/home/lonian/mamba/model/${NAME}_results/${EPOCH}_${Dset}/audio
    done
done