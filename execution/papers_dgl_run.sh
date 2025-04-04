cd .. || exit
num_layers_list=(2 3 4)
method_list=('sage')
dropout_list=(0.2)
# base command
base_command="CUDA_VISIBLE_DEVICES=0 python main_MPGNN.py --dataset ogbn-papers100M --metric acc --patience 200 --data_dir /scratch/graph_data/ --device 0 --mode mixed --test_start_epoch 29 --eval_step 10 --save_result --epochs 101 --runs 5 --plot_curve --save_result --batch_size 8000 --discrete --sampler LABOR --eval_batch_size 1000"
for method in "${method_list[@]}"
do
    if [ "$method" == "sage" ]; then
        command="$base_command --method sage --hidden_channels 256"
    else
        command="$base_command --method gat --hidden_channels 256 --num_heads 1"
    fi
    
    
    for dropout in "${dropout_list[@]}"
    do
        for num_layers in "${num_layers_list[@]}"
        do
            if [ "$num_layers" -eq 1 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 --lr 0.001"
                fi
            elif [ "$num_layers" -eq 2 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 10 --sample_sizes_eval 20 20 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 10 --lr 0.001"
                fi
            elif [ "$num_layers" -eq 3 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 10 5 --sample_sizes_eval 20 20 20 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 10 10 --lr 0.001"
                fi
            elif [ "$num_layers" -eq 4 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 10 5 3 --sample_sizes_eval 20 20 20 20 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 10 10 5 --lr 0.001"
                fi
            elif [ "$num_layers" -eq 5 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 10 5 3 3 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 10 10 5 5 --lr 0.001"
                fi
            elif [ "$num_layers" -eq 6 ]; then
                if [ "$method" == "sage" ]; then
                    eval_command="$command --sample_sizes 15 10 5 3 3 3 --lr 0.001"
                else
                    eval_command="$command --sample_sizes 10 10 10 5 5 5 --lr 0.001"
                fi
            fi

            
            echo "Running experiment for $dataset with $method and $num_layers layers"
            e_command="$eval_command --num_layers $num_layers --dropout $dropout"
            echo $e_command
            # Execute the command
            eval $e_command
        done
    done
done
echo "All experiments completed."