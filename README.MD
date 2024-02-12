
'''
path=output/1126/sess_FR_text_FR/alter_sepe_opt/hard_neg/dist_${start_rank}_${end_rank}
python -u main_hard_neg_alter_sepe_opt.py   --data_name amazonm2/sess_FR_text_FR  --seed 2 --beta1 ${beta1} --beta2 ${beta2}  --lr_dc_step_id 2 --lr_dc_step_text 5 --gap_epoch 4 --max_len 50 --start_pos_rank ${start_pos_rank} --end_pos_rank ${end_pos_rank} --aug_loss_ratio ${aug_loss_ratio} --start_rank ${start_rank} --end_rank ${end_rank} --lr ${lr} --dropout ${dropout} --l2 ${l2} --device ${device}   --lm_hidden_channel 512 --id_hidden_channel ${id_hidden_channel} --epoch 30 --output_dir ${path}  > ${path}/lr${lr}_dp${dropout}_l2${l2}_scoregnn${beta1}text${beta2}_neg_gnnrandomtraintext_textrandomtraingnn_lrid2_lrtext5_gapepoch4_seed2.out
'''