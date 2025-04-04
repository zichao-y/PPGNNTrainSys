#!/bin/bash
cd .. || exit
# Define the list of batch sizes
batch_sizes=(8000)
mode_list=('mixed')
method_list=('sage')
sampler_list=('LABOR')
num_layers_list=(2 3)
dropout_list=(0.1)

base_command="CUDA_VISIBLE_DEVICES=1 python main_MPGNN.py --dataset igb --method sage --data_dir /work/zhang-capra/users/zy383/graph_data/ --hidden_channels 256 --device 0 --test_start_epoch 10 --eval_step 10 --epochs 1 --runs 5 --num_layers 3  --metric acc --save_result --batch_size 8000 --path /work/zhang-capra/users/zy383/graph_data/IGB/ --dataset_size medium"

for sampler in "${sampler_list[@]}"
do
    for method in "${method_list[@]}"
    do  
        # base_command=$mp_command
        for mode in "${mode_list[@]}"
        do  
            for dropout in "${dropout_list[@]}"
            do 
                for num_layers in "${num_layers_list[@]}"
                do
                    if [ $num_layers -eq 2 ]
                    then
                        if [ $method == 'sage' ]
                        then
                            command="$base_command --sample_sizes 15 10"
                        else
                            command="$base_command --sample_sizes 10 10"
                        fi
                    elif [ $num_layers -eq 3 ]
                    then
                        if [ $method == 'sage' ]
                        then
                            command="$base_command --sample_sizes 15 10 5"
                        else
                            command="$base_command --sample_sizes 10 10 10"
                        fi
                    elif [ $num_layers -eq 4 ]
                    then
                        if [ $method == 'sage' ]
                        then
                            command="$base_command --sample_sizes 15 10 5 3"
                        else
                            command="$base_command --sample_sizes 10 10 10 5"
                        fi
                    elif [ $num_layers -eq 5 ]
                    then
                        if [ $method == 'sage' ]
                        then
                            command="$base_command --sample_sizes 15 10 5 3 3"
                        else
                            command="$base_command --sample_sizes 10 10 10 5 5"
                        fi
                    elif [ $num_layers -eq 6 ]
                    then
                        if [ $method == 'sage' ]
                        then
                            command="$base_command --sample_sizes 15 10 5 3 3 3"
                        else
                            command="$base_command --sample_sizes 10 10 10 5 5 5"
                        fi
                    fi
                    echo "Running $method experiment with num_layers: $num_layers mode: $mode on igb-medium"
                    
                    eval_command="$command --mode $mode --num_layers $num_layers --batch_size 8000 --sampler $sampler --dropout $dropout"
                    
                    # Print the command
                    echo $eval_command
                    # Execute the command
                    eval $eval_command
                    
                    # Optionally, add a delay or a manual check here
                    # sleep 10
                done
            done
        done
    done
done




echo "All experiments completed."
