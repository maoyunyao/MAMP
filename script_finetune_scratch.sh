export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3

# NTU-60 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 13341 main_finetune.py \
--config ./config/ntu60_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xsub_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0

# NTU-60 xview
python -m torch.distributed.launch --nproc_per_node=2 --master_port 13341 main_finetune.py \
--config ./config/ntu60_xview_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu60_xview_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0

# NTU-120 xsub
python -m torch.distributed.launch --nproc_per_node=2 --master_port 13341 main_finetune.py \
--config ./config/ntu120_xsub_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xsub_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xsub_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0

# NTU-120 xset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 13341 main_finetune.py \
--config ./config/ntu120_xset_joint/finetune_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu120_xset_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--log_dir ./output_dir/ntu120_xset_joint/finetune_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5 \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0