cd .. || exit
num_layers_list=(2 3 4 5 6)
method_list=('sage' 'gat')
sampler_list=('LABOR' 'neighbor')
# base command
base_command="CUDA_VISIBLE_DEVICES=1 python main_MPGNN.py --metric acc --dataset wiki --patience 200 --data_dir /scratch/graph_data/ --device 0 --mode puregpu --eval_step 10 --save_result --test_start_epoch -1 --epochs 400 --runs 5 --save_result --batch_size 8000 --eval_batch_size 100000"
for method in "${method_list[@]}"
do
    if [ "$method" == "sage" ]; then
        command="$base_command --method sage --hidden_channels 256"
    else
        command="$base_command --method gat --hidden_channels 128 --num_heads 4"
    fi
        
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
                eval_command="$command --sample_sizes 15 10 --lr 0.001"
            else
                eval_command="$command --sample_sizes 10 10 --lr 0.001"
            fi
        elif [ "$num_layers" -eq 3 ]; then
            if [ "$method" == "sage" ]; then
                eval_command="$command --sample_sizes 15 10 5 --lr 0.001"
            else
                eval_command="$command --sample_sizes 10 10 10 --lr 0.001"
            fi
        elif [ "$num_layers" -eq 4 ]; then
            if [ "$method" == "sage" ]; then
                eval_command="$command --sample_sizes 15 10 5 3 --lr 0.001"
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

        for sampler in "${sampler_list[@]}"
        do
            if [ "$sampler" == "LABOR" ]; then
                dropout=0.2
            else
                if [ "$num_layers" -eq 5 ]; then
                    dropout=0.2
                elif [ "$num_layers" -eq 6 ]; then
                    dropout=0.1
                else
                  dropout=0.5
                fi
            fi
            echo "Running experiment for wiki with $method and $num_layers layers using $sampler sampler"
            e_command="$eval_command --num_layers $num_layers --dropout $dropout --sampler $sampler"
            # Execute the command
            eval $e_command
        done
    done
done
echo "All experiments completed."