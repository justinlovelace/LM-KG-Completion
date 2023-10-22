gpu_id=0

CUDA_VISIBLE_DEVICES="$gpu_id" python train.py --strategy one_to_n --clip 1 --batch_size 64 --dropout 0.3 --feature_map_dropout 0.2 --input_dropout 0.2 --weight_decay 0.0001 --lr 0.001 --dataset CN82K --model PretrainedBertResNet --reshape_len 5 --resnet_block_depth 2 --resnet_num_blocks 3 --label_smoothing_epsilon 0.1 --num_epochs 200 --head_bert_model bert-base-uncased --save_dir saved_models/tail_experiment --head_bert_pool prior --tail_lr 1e-4 --tail_bert_model bert-base-uncased --tail_bert_pool prior --tail_embed res_mlp --run_id res_mlp
CUDA_VISIBLE_DEVICES="$gpu_id" python evaluation.py --model PretrainedBertResNet --dataset CN82K --save_dir saved_models/tail_experiment --model_folder res_mlp
