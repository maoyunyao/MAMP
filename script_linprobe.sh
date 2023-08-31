export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1


# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
--log_dir ./output_dir/ntu60_xsub_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_400 \
--log_dir ./output_dir/ntu60_xsub_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_400 \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval



# NTU-60 xview
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xview_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xview_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
--log_dir ./output_dir/ntu60_xview_joint/linear_mae_t120_layer8+3_mask90_ep400_400 \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu60_xview_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu60_xview_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_400 \
--log_dir ./output_dir/ntu60_xview_joint/linear_mamp_t120_layer8+3_mask90_tau0.80_ep400_400 \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval



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



# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu120_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_400 \
--log_dir ./output_dir/ntu120_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_400 \
--finetune ./output_dir/ntu120_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval

# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/ntu120_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.75_ep400_400 \
--log_dir ./output_dir/ntu120_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.75_ep400_400 \
--finetune ./output_dir/ntu120_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.75_ep400_noamp/checkpoint-399.pth \
--dist_eval



# PKU Phase I
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv1_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv1_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_300 \
--log_dir ./output_dir/pkuv1_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_300 \
--finetune ./output_dir/pkuv1_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-300.pth \
--dist_eval

# PKU Phase I
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv1_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv1_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.80_ep400_300 \
--log_dir ./output_dir/pkuv1_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.80_ep400_300 \
--finetune ./output_dir/pkuv1_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp/checkpoint-300.pth \
--dist_eval



# PKU Phase I
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv2_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_300 \
--log_dir ./output_dir/pkuv2_xsub_joint/linear_mae_t120_layer8+5_mask90_ep400_300 \
--finetune ./output_dir/pkuv2_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-300.pth \
--dist_eval

# PKU Phase I
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_linprobe.py \
--config ./config/pkuv2_xsub_joint/linprobe_t120_layer8.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.80_ep400_300 \
--log_dir ./output_dir/pkuv2_xsub_joint/linear_mamp_t120_layer8+5_mask90_tau0.80_ep400_300 \
--finetune ./output_dir/pkuv2_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.80_ep400_noamp/checkpoint-300.pth \
--dist_eval