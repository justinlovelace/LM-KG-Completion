gpu_id=0

for bert_model in bert-base-uncased-ft
do 
    for bert_pool in mean
    do
        echo "$bert_pool";
        echo "$bert_model";
        CUDA_VISIBLE_DEVICES="$gpu_id" python train.py --dataset CN82K --bert_model "$bert_model" --bert_pool "$bert_pool" --batch_size 64 --lr 1e-3 --num_epochs 500 --max_grad_norm 1
    done
done

