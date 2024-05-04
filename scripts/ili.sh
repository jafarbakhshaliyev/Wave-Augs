
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/onerun" ]; then
    mkdir ./logs/onerun
fi

seq_len=36
percentage=100
model_name=DLinear

# aug 0: None
# aug 1: Frequency Masking 
# aug 2: Frequency Mixing
# aug 3: Wave Masking
# aug 4: Wave Mixing
# aug 5: STAug

pred_lens=(24 36 48 60)



# For Aug 0: None

for pred_len in "${pred_lens[@]}"; do
    python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data  custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 0 --aug_rate 0.0 >logs/onerun/$model_name'_'ill_$seq_len'_'$pred_len'_'0.0'_'$percentage'_'None.log
done


# For Aug 1: Freq-Masking


python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len  24 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 1 --aug_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'96'_'0.4'_'$percentage'_'FreqMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len  36 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 1 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'36'_'0.4'_'$percentage'_'FreqMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len  48 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 1 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'48'_'0.4'_'$percentage'_'FreqMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len  60 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 1 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'60'_'0.4'_'$percentage'_'FreqMask.log



# For Aug 2: Freq-Mixing

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 24 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 2 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'24'_'0.4'_'$percentage'_'FreqMix.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 36 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 2 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'36'_'0.4'_'$percentage'_'FreqMix.log


python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 48 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 2 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'48'_'0.4'_'$percentage'_'FreqMix.log


python3 -u ../run_main2.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 60 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 2 --aug_rate 0.1 >logs/onerun/$model_name'_'ill_$seq_len'_'60'_'0.4'_'$percentage'_'FreqMix.log



# For Aug 3: Wave Masking


python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 24 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 3 --rates "[0.4, 0.8, 0.9, 0.7, 0.9, 0.0, 0.5]" --wavelet 'db25' --level 1 --mask_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'24'_'0.0'_'$percentage'_'WaveMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 36 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 3 --rates "[0.6, 0.8, 0.3, 0.1, 0.9, 0.0, 0.5]" --wavelet 'db25' --level 1 --mask_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'36'_'0.0'_'$percentage'_'WaveMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 48 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 3 --rates "[0.2, 0.7, 1.0, 0.4, 0.4, 0.0, 0.5]" --wavelet 'db2' --level 1 --mask_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'48'_'0.0'_'$percentage'_'WaveMask.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 60 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 3 --rates "[0.2, 0.8, 0.5, 0.1, 0.9, 0.0, 0.5]" --wavelet 'db25' --level 1 --mask_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'60'_'0.0'_'$percentage'_'WaveMask.log

# For Aug 4: Wave Mixing

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 24 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 4 --rates "[0.1, 0.8, 1.0, 0.0, 0.5, 0.7, 0.1]" --wavelet 'db1' --level 1 --mask_rate 0.2 >logs/onerun/$model_name'_'ill_$seq_len'_'24'_'0.0'_'$percentage'_'WaveMix.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 36 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 4 --rates "[0.1, 1.0, 0.9, 0.2, 0.1, 0.7, 0.1]" --wavelet 'db25' --level 1 --mask_rate 0.8 >logs/onerun/$model_name'_'ill_$seq_len'_'36'_'0.0'_'$percentage'_'WaveMix.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 48 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 4 --rates "[0.1, 1.0, 0.4, 0.5, 0.1, 0.6, 0.1]" --wavelet 'db3' --level 1 --mask_rate 1.0 >logs/onerun/$model_name'_'ill_$seq_len'_'48'_'0.0'_'$percentage'_'WaveMix.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv  \
    --model $model_name \
    --data  custom  \
    --features M \
    --seq_len $seq_len \
    --pred_len 60 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 4 --rates "[0.1, 0.9, 0.3, 0.9, 0.5, 0.7, 0.1]" --wavelet 'db1' --level 1 --mask_rate 0.5 >logs/onerun/$model_name'_'ill_$seq_len'_'60'_'0.0'_'$percentage'_'WaveMix.log


# For Aug 5: STAug
    
python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data  custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 24 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 5 --aug_rate 0.7 --nIMF 200 >logs/onerun/$model_name'_'ill_$seq_len'_'24'_'0.9'_'$percentage'_'StAug.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data  custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 36 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 5 --aug_rate 0.3 --nIMF 300 >logs/onerun/$model_name'_'ill_$seq_len'_'36'_'0.9'_'$percentage'_'StAug.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data  custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 48 \
    --enc_in 7 \
    --des '100p-ili-' \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 5 --aug_rate 0.9 --nIMF 300 >logs/onerun/$model_name'_'ill_$seq_len'_'48'_'0.9'_'$percentage'_'StAug.log

python3 -u ../run_main.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data  custom \
    --features M \
    --seq_len $seq_len \
    --pred_len 60 \
    --enc_in 7 \
    --des '100p-ili-'  \
    --percentage $percentage \
    --itr 10 --batch_size 32 --learning_rate 0.01 --aug_type 5 --aug_rate 0.7 --nIMF 1000 >logs/onerun/$model_name'_'ill_$seq_len'_'60'_'0.9'_'$percentage'_'StAug.log

