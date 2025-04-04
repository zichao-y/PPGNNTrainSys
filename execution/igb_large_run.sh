cd .. || exit
# Define the list of training hops (training_hops and testing_hops will be the same)
training_hops_list=(2 3)
chunk_list=(1)
method_list=(HOGA SIGN)

base_command_HOGA="CUDA_VISIBLE_DEVICES=0 python main.py --method HOGA --dataset igb --dataset_size large --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 10 --data_dir /work/zhang-capra/users/zy383/graph_data/ --lr 0.001 --weight_decay 0 --num_heads 4 --hidden_channels 256 --mlplayers 2 --dropout 0.1 --attn_dropout 0.0 --input_dropout 0.0 --eval_batch --use_post_res 1 --input_type da --num_layers 1 --save_result --batch_size 8000 --load_all --mode gpu --save_result --chunk_size 1"

base_command_SIGN="CUDA_VISIBLE_DEVICES=0 python main.py --method SIGN --dataset igb --dataset_size large --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 10 --data_dir /work/zhang-capra/users/zy383/graph_data/ --lr 0.001 --weight_decay 0  --hidden_channels 512  --dropout 0.15 --eval_batch --use_post_res 1 --input_type da --num_layers 3 --save_result --batch_size 8000 --cat_input --load_all --mode gpu --save_result --chunk_size 1"

base_command_SGC="CUDA_VISIBLE_DEVICES=0 python main.py --method SGC --dataset igb --dataset_size large --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 10 --data_dir /scratch/graph_data/  --lr 0.001 --weight_decay 0  --hidden_channels 512 --dropout 0.5 --eval_batch --input_type da --save_result --batch_size 8000 --load_all --save_result --mode gpu"


for training_hops in "${training_hops_list[@]}"
do
    for method in "${method_list[@]}"
    do
        if [ $method == 'SIGN' ]
        then
            command=$base_command_SIGN
        elif [ $method == 'HOGA' ]
        then
            command=$base_command_HOGA
        elif [ $method == 'SGC' ]
        then
            command=$base_command_SGC
        fi
        
        for chunk_size in "${chunk_list[@]}"
        do
            echo "Running experiment with training/testing hops: $training_hops, method: $method and chunk_size: $chunk_size"
            eval_command="$command --training_hops $training_hops --testing_hops $training_hops --dropout $dropout"
            echo $eval_command
            # Execute the command
            eval $eval_command
        done
    done
done


echo "All experiments completed."