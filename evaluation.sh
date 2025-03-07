# step 1. data preparation
# Can set block_size to a non-zero value to enable blocked data preprocessing for large datasets, e.g., --block_size 100000
python preprocess.py --dataset ogbn-products --data_dir ./data/ --save_dir ./data/ --training_hops 3

# Single GPU evaluation
# step 2. training with PPGNN baseline implementation
python main_baseline.py --model_config model_cfg.json --pipeline_config pipeline_cfg.json --save_json
# step 3. training with GPU memory preloading + SGDRR + Double buffering, save accuracy results of 5 runs with 400 epochs
python main.py --model_config model_cfg.json --pipeline_config ./policy/preload_all.json --save_json
# step 4. training with Data residing in host main memory + SGDRR + GPU-side Double buffering
python main.py --model_config model_cfg.json --pipeline_config ./policy/uvm_RR.json --save_json
# step 5. training with Data residing in host main memory + SGDCR + GPU-side Double buffering
python main.py --model_config model_cfg.json --pipeline_config ./policy/uvm_CR.json --save_json
# step 6. training with GDS + SGDRR + GPU-side Double buffering
python main.py --model_config model_cfg.json --pipeline_config ./policy/gds.json --save_json


# Multi-GPU evaluation, change the gpu_ids to the GPU devices you want to use
# step 7. training with GPU memory preloading + SGDRR + Double buffering
python main_mp.py --model_config model_cfg.json --pipeline_config ./policy/preload_all.json --gpu_ids 0 1  --save_json
# step 8. training with Data residing in host main memory + SGDRR + GPU-side Double buffering
python main_mp.py --model_config model_cfg.json --pipeline_config ./policy/uvm_RR.json --gpu_ids 0 1 --save_json
# step 9. training with Data residing in host main memory + SGDCR + GPU-side Double buffering
python main_mp.py --model_config model_cfg.json --pipeline_config ./policy/uvm_CR.json --gpu_ids 0 1 --save_json

# Automated training pipeline configuration 
# add --GPUcap N to constrain the number of GPUs to be used to no larger than N
# save the accuracy results of 5 runs with 400 epochs
python auto_run.py --model_config model_acc_cfg.json --save_result