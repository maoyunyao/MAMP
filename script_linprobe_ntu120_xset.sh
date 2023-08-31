export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3


# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu120_xset_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xset_joint/linear_mae_t120_layer8+5_mask90_ep400_400 \
--log_dir ./output_dir/ntu120_xset_joint/linear_mae_t120_layer8+5_mask90_ep400_400 \
--finetune ./output_dir/ntu120_xset_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval

# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu120_xset_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xset_joint/linear_mamp_t120_layer8+5_mask90_tau0.75_ep400_400 \
--log_dir ./output_dir/ntu120_xset_joint/linear_mamp_t120_layer8+5_mask90_tau0.75_ep400_400 \
--finetune ./output_dir/ntu120_xset_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.75_ep400_noamp/checkpoint-399.pth \
--dist_eval