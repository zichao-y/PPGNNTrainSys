cd .. || exit
num_layers_list=(2 3 4 5 6)
method_list=('sage' 'gat')
dropout_list=(0.5)
sampler_list=('LABOR' 'neighbor')
# base command
base_command="CUDA_VISIBLE_DEVICES=1 python main_MPGNN.py --dataset ogbn-products --patience 1000 --metric acc --data_dir /scratch/graph_data/ --device 0 --mode puregpu --eval_step 10 --test_start_epoch 100 --save_result --epochs 10 --runs 5 --lr 0.001 --batch_size 8000 "
for method in "${method_list[@]}"
do
    if [ "$method" == "sage" ]; then
        base_command="$base_command --method sage --hidden_channels 256"
    else
        base_command="$base_command --method gat --hidden_channels 128 --num_heads 4"
    fi
        
    for dropout in "${dropout_list[@]}"
    do
        for num_layers in "${num_layers_list[@]}"
        do
            if [ "$num_layers" -eq 1 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 "
                else
                    command="$base_command --sample_sizes 10"
                fi
            elif [ "$num_layers" -eq 2 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 10"
                else
                    command="$base_command --sample_sizes 10 10"
                fi
            elif [ "$num_layers" -eq 3 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 10 5"
                else
                    command="$base_command --sample_sizes 10 10 10"
                fi
            elif [ "$num_layers" -eq 4 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 10 5 3"
                else
                    command="$base_command --sample_sizes 10 10 10 5"
                fi
            elif [ "$num_layers" -eq 5 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 10 5 3 3"
                else
                    command="$base_command --sample_sizes 10 10 10 5 5"
                fi
            elif [ "$num_layers" -eq 6 ]; then
                if [ "$method" == "sage" ]; then
                    command="$base_command --sample_sizes 15 10 5 3 3 3"
                else
                    command="$base_command --sample_sizes 10 10 10 5 5 5"
                fi
            fi

            if [ "$method" == "gat" ] && [ "$num_layers" -eq 5 ]; then
                dropout=0.3
            elif [ "$method" == "gat" ] && [ "$num_layers" -eq 6 ]; then
                dropout=0.2
            fi
            for sampler in "${sampler_list[@]}"
            do
                eval_command="$command --num_layers $num_layers --dropout $dropout --sampler $sampler"
                echo $eval_command
                # Execute the command
                eval $eval_command
            done
        done
    done
done
echo "All experiments completed."