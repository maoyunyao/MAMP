export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3


######## NTU60 xsub
# mamp
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.01_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.01_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.1_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.1_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done



# mae
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.01_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.01_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.1_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.1_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xsub_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done




# scratch
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.01_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.01_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xsub_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xsub_joint/semi_0.1_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xsub_joint/semi_0.1_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0
done








######## NTU60 xview
# mamp
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.01_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.01_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.1_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.1_mamp_t120_layer8+3_mask90_tau0.80_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done



# mae
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.01_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.01_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.1_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.1_mae_t120_layer8+3_mask90_ep400_400_warm5_dpr0.3_dr0.3_decay0.8_lr3e-4_minlr1e-5_{$i} \
--finetune ./output_dir/ntu60_xview_joint/pretrain_mae_t120_layer8+3_mask90_ep400_noamp/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5
done



# scratch
for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.01_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.01_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.01_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0
done

for((i=1;i<=5;i++)); 
do
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60_xview_joint/semi_0.1_t120_layer8_decay.yaml \
--output_dir ./output_dir/ntu60_xview_joint/semi_0.1_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--log_dir ./output_dir/ntu60_xview_joint/semi_0.1_scratch_t120_layer8_warm5_dpr0.3_dr0.3_decay1.0_lr3e-4_minlr1e-5_{$i} \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0
done