export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7


# NTU60 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp \
--log_dir ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp

# NTU60 xview
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp \
--log_dir ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp

# NTU120 xset
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu120_xset_joint/pretrain_mae_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/ntu120_xset_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp \
--log_dir ./output_dir/ntu120_xset_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp

# NTU120 xsub
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu120_xsub_joint/pretrain_mae_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp \
--log_dir ./output_dir/ntu120_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp

# PKU v1
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/pkuv1_xsub_joint/pretrain_mae_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/pkuv1_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp \
--log_dir ./output_dir/pkuv1_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp

# PKU v2
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/pkuv2_xsub_joint/pretrain_mae_t120_layer8+5_mask90.yaml \
--output_dir ./output_dir/pkuv2_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp \
--log_dir ./output_dir/pkuv2_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp