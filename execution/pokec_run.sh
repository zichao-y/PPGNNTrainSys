cd .. || exit
# Define the list of training hops (training_hops and testing_hops will be the same)
training_hops_list=(2 3 4 5 6)
method_list=(HOGA SIGN SGC)
# base command
base_command_HOGA="CUDA_VISIBLE_DEVICES=0 python main.py --method HOGA --dataset pokec --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 400 --data_dir /scratch/graph_data/ --lr 0.001 --num_heads 1 --hidden_channels 256 --mlplayers 2 --attn_dropout 0.0 --input_dropout 0.0 --eval_batch --use_post_res 1 --input_type da --num_layers 1 --save_result --batch_size 8000 --load_all --mode gpu --save_result --chunk_size 1"
base_command_SIGN="CUDA_VISIBLE_DEVICES=1 python main.py --method MLP --dataset pokec --device 0 --runs 5 --test_start_epoch -1 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 400 --data_dir /scratch/graph_data/ --lr 0.001  --hidden_channels 512  --dropout 0.5 --eval_batch  --input_type da --num_layers 3 --save_result --batch_size 8000  --cat_input --load_all --mode gpu --save_result --chunk_size 1"
base_command_SGC="CUDA_VISIBLE_DEVICES=0 python main.py --method SGC --dataset pokec --device 0 --runs 5 --test_start_epoch 20 --eval_step 10 --metric acc --patience 1000 --weight_decay 0.0 --epochs 400 --data_dir /scratch/graph_data/  --lr 0.001 --hidden_channels 512  --dropout 0.15 --eval_batch  --input_type da --save_result --batch_size 8000  --load_all --mode gpu --save_result --chunk_size 1"
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
    for training_hops in "${training_hops_list[@]}"
    do  
        if [ $training_hops -eq 5 ] && [ $method == 'HOGA' ]
        then
            dropout=0.1
        elif [ $training_hops -eq 6 ] && [ $method == 'HOGA' ]
        then
            dropout=0.2
        elif [ $training_hops -eq 2 ] && [ $method == 'SIGN' ]
        then
            dropout=0.5
        elif [ $training_hops -eq 3 ] && [ $method == 'SIGN' ]
        then
            dropout=0.5
        elif [ $method == 'SIGN' ]
        then
            dropout=0.8
        elif [ $method == 'HOGA' ]
        then
            dropout=0.3
        else
            dropout=0.5
        fi
        eval_command="$command --training_hops $training_hops --testing_hops $training_hops --dropout $dropout --save_model"
        echo $eval_command
        # Execute the command
        eval $eval_command    
    done
done
echo "All experiments completed."