from parse import load_args
import json
import psutil
import pynvml
import time
import subprocess
import os
import torch

class AutoConfigSystem:
    def __init__(self):
        self.args = load_args()
        self.system_config = self.get_system_config()
        self.gpu_info = self.get_gpu_memory_info()
        self.profile_data = None
        self._init_profile_vars()
        
    def _init_profile_vars(self):
        """Initialize profile-dependent variables"""
        self.peak_host_mem = None
        self.peak_gpu_mem = None
        self.number_nodes = None
        self.in_dim = None
        self.train_len = None
        self.data_size = None
        self.train_size = None

    def get_system_config(self):
        """Gather system hardware configuration"""
        mem_info = psutil.virtual_memory()
        return {'total_host_memory': mem_info.total}

    def get_gpu_memory_info(self):
        """Collect GPU memory information"""
        pynvml.nvmlInit()
        gpu_info = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.append({
                'device_id': i,
                'name': pynvml.nvmlDeviceGetName(handle),
                'total_memory': meminfo.total,
                'free_memory': meminfo.free
            })
        pynvml.nvmlShutdown()
        return gpu_info

    def execute_policy(self, policy_name, json_path=None, device_ids=None, chunk_size=None, print_output=True):
        """Execute a training policy with proper resource allocation"""
        command = [
            "python", "main_mp.py" if device_ids else "main.py",
            "--dataset", self.args.dataset,
            "--sub_dataset", self.args.sub_dataset,
            "--data_dir", self.args.data_dir,
            "--model_config", self.args.model_config,
            "--pipeline_config", f"policy/{policy_name}.json",
        ]
        
        if json_path:
            command += ["--sys_json", json_path]
        if device_ids:
            command += ["--gpu_ids"] + [str(d) for d in device_ids]
        if chunk_size:
            command += ["--chunk_size", str(chunk_size)]
        if self.args.save_result:
            command += ["--save_result"]
        policy_prefix="SingleGPU" if not device_ids else f"{len(device_ids)}_GPUs"
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in {policy_prefix}_{policy_name}: {result.stderr}")
            return False
        else:
            if print_output:
                print(result.stdout)
            print(f"Success with {policy_prefix}_{policy_name}")
        return True

    def run_profiling(self):
        """Run initial profiling trial to gather resource requirements"""
        json_path = f'temp_{time.strftime("%Y-%m-%d_%H:%M:%S")}.json'
        if not self.execute_policy("trail", json_path=json_path, print_output=False):
            return False
            
        with open(json_path, 'r') as f:
            self.profile_data = json.load(f)
            
        # Set profile-dependent variables
        self.peak_host_mem = self.profile_data['peak_host_mem']
        self.peak_gpu_mem = self.profile_data['peak_gpu_mem']
        self.number_nodes = self.profile_data['number_nodes']
        self.in_dim = self.profile_data['in_dim']
        self.train_len = self.profile_data['num_train_nodes']
        
        if self.args.input_type == 'da_dad':
            self.in_dim *= 2
            
        self.data_size = self.number_nodes * self.in_dim * 4 * (self.args.training_hops + 1)
        self.train_size = self.train_len * self.in_dim * 4 * (self.args.training_hops + 1)
        return True

    def run_single_gpu_policy(self):
        """Handle single GPU policy selection"""
        # Select GPU with maximum free memory
        best_gpu = max(self.gpu_info, key=lambda x: x['free_memory'])
        gpu_id = best_gpu['device_id']
        # Set environment variable for CUDA device selection
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Need to re-initialize CUDA context after this change
        torch.cuda.init() 
        if self.train_size + self.peak_gpu_mem > self.max_gpu_mem:
            if self.train_size > self.host_available_mem * self.host_memory_pin_threshold:
                return self.execute_policy("uvm_RR")
            return self.execute_policy("uvm_CR", chunk_size=self.args.batch_size)
            
        if self.execute_policy("preload_all"):
            return True
        return self.execute_policy("preload_train")

    def run_multi_gpu_policy(self):
        """Handle multi-GPU policy selection with fallback"""
        valid_gpus = sorted(
            [g for g in self.gpu_info if g['free_memory'] > self.peak_gpu_mem],
            key=lambda x: -x['free_memory']
        )
        
        max_gpus = min(
            int(self.host_available_mem // self.peak_host_mem),
            len(valid_gpus),
            self.args.GPUcap
        )
        
        for gpu_count in range(max_gpus, 0, -1):
            if gpu_count == 1:
                return self.run_single_gpu_policy()
                
            current_gpus = valid_gpus[:gpu_count]
            device_ids = [g['device_id'] for g in current_gpus]
            
            for policy in ['preload_all', 'preload_train',
                          'uvm_CR', 'uvm_RR']:
                if self.execute_policy(policy, device_ids=device_ids):
                    return True
        return False

    @property
    def host_available_mem(self):
        return self.system_config['total_host_memory']
        
    @property
    def max_gpu_mem(self):
        return max(g['free_memory'] for g in self.gpu_info)
        
    @property
    def host_memory_pin_threshold(self):
        return self.args.host_mem_thr[2]

    def run(self):
        """Main execution flow"""
        if not self.run_profiling():
            print("Initial profiling failed")
            return False
        
        if not self.gpu_info:
            print("No available GPUs for training!")
            return self.execute_policy("cpu")
        
        if self.data_size > self.host_available_mem * self.args.host_mem_thr[1]:
            return self.execute_policy("gds", chunk_size=self.args.batch_size)
            
        if len(self.gpu_info) > 1 and self.args.GPUcap > 1:
            return self.run_multi_gpu_policy()
        return self.run_single_gpu_policy()

if __name__ == "__main__":
    configurator = AutoConfigSystem()
    success = configurator.run()
    # clean up all the temp_***.json files
    for f in os.listdir():
        if f.startswith('temp_') and f.endswith('.json'):
            os.remove(f)
    exit(0 if success else 1)