cd .. || exit
num_layers_list=(2 3 4 5 6)
method_list=('sage' 'gat')
sampler_list=('LABOR' 'neighbor')
# base command
base_command="CUDA_VISIBLE_DEVICES=1 python main_MPGNN.py --dataset pokec --patience 400 --metric acc --data_dir /scratch/graph_data/  --device 0 --mode puregpu --eval_step 1 --test_start_epoch -1 --save_result --epochs 400 --runs 1 --plot_curve --save_result --lr 0.001 --batch_size 8000"
for method in "${method_list[@]}"
do
    if [ "$method" == "sage" ]; then
        base_command="$base_command --method sage --hidden_channels 256"
    else
        base_command="$base_command --method gat --hidden_channels 128 --num_heads 4"
    fi
    for num_layers in "${num_layers_list[@]}"
    do
        if [ "$num_layers" -eq 1 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 --dropout 0.2"
            else
                command="$base_command --sample_sizes 10 --dropout 0.2"
            fi
        elif [ "$num_layers" -eq 2 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 10 --dropout 0.2"
            else
                command="$base_command --sample_sizes 10 10 --dropout 0.2"
            fi
        elif [ "$num_layers" -eq 3 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 10 5 --dropout 0.3"
            else
                command="$base_command --sample_sizes 10 10 10 --dropout 0.2"
            fi
        elif [ "$num_layers" -eq 4 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 10 5 3 --dropout 0.2"
            else
                command="$base_command --sample_sizes 10 10 10 5 --dropout 0.2"
            fi
        elif [ "$num_layers" -eq 5 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 10 5 3 3 --dropout 0.2"
            else
                command="$base_command --sample_sizes 10 10 10 5 5 --dropout 0.2"
            fi
        elif [ "$num_layers" -eq 6 ]; then
            if [ "$method" == "sage" ]; then
                command="$base_command --sample_sizes 15 10 5 3 3 3 --dropout 0.3"
            else
                command="$base_command --sample_sizes 10 10 10 5 5 5 --dropout 0.2"
            fi
        fi

        for sampler in "${sampler_list[@]}"
        do
            echo "Running experiment for pokec with $method and $num_layers layers using $sampler sampler"
            eval_command="$command --num_layers $num_layers --sampler $sampler"
            echo $eval_command
            # Execute the command
            eval $eval_command
        done
    done
done
echo "All experiments completed."