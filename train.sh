# Create a symbolic link for the /projects/0/prjs0839/data/MM-PCQA/ directory
# ln -s /home/xzhou/data /projects/0/prjs0839/data/MM-PCQA/
# python -u train_img.py \
# --learning_rate 0.00005 \
# --model ViSam_PCQA \
# --batch_size  18 \
# --database SJTU  \
# --data_dir_texture_img /projects/0/prjs0839/data/MM-PCQA/sjtu_projections_xm/ \
# --data_dir_depth_img /projects/0/prjs0839/data/MM-PCQA/sjtu_depth_maps/ \
# --data_dir_normal_img /projects/0/prjs0839/data/MM-PCQA/sjtu_normal_maps/ \
# --data_dir_vs_img /projects/0/prjs0839/data/MM-PCQA/sjtu_vs/ \
# --loss l2rank \
# --num_epochs 100 \
# --k_fold_num 9 \
# --use_classificaiton 1 \
# --use_local 1 \
# --method_label Baseline

# python -u train_img.py \
# --learning_rate 0.00005 \
# --model ViSam_PCQA \
# --batch_size 18 \
# --database WPC  \
# --data_dir_texture_img /projects/0/prjs0839/data/MM-PCQA/wpc_projections_xm/ \
# --data_dir_depth_img /projects/0/prjs0839/data/MM-PCQA/wpc_depth_maps/ \
# --data_dir_normal_img /projects/0/prjs0839/data/MM-PCQA/wpc_normal_maps/ \
# --data_dir_vs_img /projects/0/prjs0839/data/MM-PCQA/wpc_vs/ \
# --loss l2rank \
# --num_epochs 100 \
# --k_fold_num 5 \
# --use_classificaiton 1 \
# --use_local 1 \
# --method_label Baseline

python -u train_img.py \
--learning_rate 0.00005 \
--model ViSam_PCQA \
--batch_size 18 \
--database BASICS  \
--data_dir_texture_img /projects/0/prjs0839/data/MM-PCQA/basic_projections_xm/ \
--data_dir_depth_img /projects/0/prjs0839/data/MM-PCQA/basic_depth_maps/ \
--data_dir_normal_img /projects/0/prjs0839/data/MM-PCQA/basic_normal_maps/ \
--data_dir_vs_img /projects/0/prjs0839/data/MM-PCQA/basic_vs/ \
--loss l2rank \
--num_epochs 100 \
--k_fold_num 1 \
--use_classificaiton 1 \
--use_local 1 \
--method_label Baseline

