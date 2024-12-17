# Create a symbolic link for the /projects/0/prjs0839/data/ directory
# ln -s /home/xzhou/data /projects/0/prjs0839/data/
python -u train_img.py \
--learning_rate 0.00005 \
--model ViSam_PCQA \
--batch_size 18 \
--database BASICS  \
--data_dir_texture_img path to basic projections/ \
--data_dir_depth_img path to basic depth_maps/ \
--data_dir_normal_img path to basic normal_maps/ \
--data_dir_vs_img path to basic basic_visual saliency maps/ \
--loss l2rank \
--num_epochs 100 \
--k_fold_num 1 \
--use_classificaiton 1 \
--use_local 1 \
--method_label Baseline

