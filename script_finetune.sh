export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6,7

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/finetune_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xsub_joint/finetune_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/finetune_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xsub_joint/finetune_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5



# NTU-60 xview
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/finetune_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xview_joint/finetune_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

# NTU-60 xview
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/finetune_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xview_joint/finetune_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5



# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu120_xset_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xset_joint/finetune_mae_t120_layer8+5_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xset_joint/finetune_mae_t120_layer8+5_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu120_xset_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu120_xset_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xset_joint/finetune_mamp_t120_layer8+5_mask90_tau0.75_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xset_joint/finetune_mamp_t120_layer8+5_mask90_tau0.75_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu120_xset_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.75_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5



# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu120_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/finetune_mae_t120_layer8+5_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xsub_joint/finetune_mae_t120_layer8+5_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu120_xsub_joint/pretrain_mae_t120_layer8+5_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu120_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/finetune_mamp_t120_layer8+5_mask90_tau0.75_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xsub_joint/finetune_mamp_t120_layer8+5_mask90_tau0.75_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5 \
--finetune ./output_dir/ntu120_xsub_joint/pretrain_mamp_t120_layer8+5_mask90_tau0.75_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5