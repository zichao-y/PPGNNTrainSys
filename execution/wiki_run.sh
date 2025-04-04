cd .. || exit
# Define the list of training hops (training_hops and testing_hops will be the same)
training_hops_list=(2 3 4 5 6)
method_list=(HOGA SIGN SGC)
# base command
base_command_HOGA="CUDA_VISIBLE_DEVICES=0 python main.py --method HOGA --dataset wiki --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 50 --weight_decay 0.0 --epochs 400 --data_dir /work/zhang-capra/users/zy383/graph_data/ --batch_size 8000 --lr 0.001 --weight_decay 0 --num_heads 1 --hidden_channels 256 --mlplayers 2 --dropout 0.15 --attn_dropout 0.0 --input_dropout 0.0 --use_post_res 1 --input_type dad --num_layers 1 --save_result --load_all --eval_batch --mode gpu --chunk_size 1"
base_command_SIGN="CUDA_VISIBLE_DEVICES=0 python main.py --method SIGN --dataset wiki --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 50 --weight_decay 0.0 --epochs 400 --data_dir /work/zhang-capra/users/zy383/graph_data/ --batch_size 8000 --lr 0.001 --weight_decay 0 --hidden_channels 512 --dropout 0.5 --input_type dad --num_layers 3 --save_result --load_all --eval_batch --cat_input --mode gpu --chunk_size 1"
base_command_SGC="CUDA_VISIBLE_DEVICES=0 python main.py --method SGC --dataset wiki --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 50 --weight_decay 0.0 --epochs 400 --data_dir /work/zhang-capra/users/zy383/graph_data/ --batch_size 8000 --lr 0.001 --weight_decay 0 --hidden_channels 512 --dropout 0.7 --input_type dad --save_result --load_all --eval_batch --mode gpu"


for method in "${method_list[@]}"
do
    if [ $method == 'SIGN' ]
    then
        command=$base_command_SIGN
    elif [ $method == 'HOGA' ]
    then
        command=$base_command_HOGA
    elif [ $method == 'sgc' ]
    then
        command=$base_command_SGC
    fi
    for training_hops in "${training_hops_list[@]}"
    do
        if [ $method == 'SIGN' ]
        then
            dropout=0.7
        elif [ $method == 'HOGA' ]
        then
            dropout=0.15
        else
            dropout=0.5
        fi
        # echo "Running experiment with training/testing hops: $training_hops and method: $method"
        eval_command="$command --training_hops $training_hops --testing_hops $training_hops --dropout $dropout --save_model"
        # Execute the command
        echo $eval_command
        eval $eval_command
    done
done

echo "All experiments completed."