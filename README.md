# **PP-GNN Training System**  

## **Installation Guide**  

### **1. Set up the Conda Environment**  
Create a Conda environment and install dependencies by executing:  

```bash
conda env create -f env.yaml
```

### **2. Install IGB**  
Download and install the Illinois Graph Benchmark (IGB) dataset:  

```bash
git clone https://github.com/IllinoisGraphBenchmark/IGB-Datasets.git
cd IGB-Datasets
pip install .
```

### **3. Install Customized Operators**  

#### **Install `async_fetch` Operator**  
```bash
cd async_fetch
pip install .
```

#### **Install `gds_read` Operator**  
> **Important:** Ensure that **NVIDIA GDS** is installed on your system.  
> Additionally, update the `CUDA_PATH` in `setup.py` to match your system's CUDA installation.  

```bash
cd gds_read
pip install .
```

---

## **Artifact Evaluation**  
We provide instructions for evaluating our work on the **ogbn-products** dataset.  
Training on other datasets follows a similar workflow. Training scripts for other datasets and the MPGNN baselines are provided under ./execution/

### **Workflow Overview**  
The evaluation script evaluation.sh consists of four main parts:

1. **Preprocessing**  
   - Convert the dataset into a format suitable for PP-GNN training.

2. **Single-GPU Experiments**  
   - Compare vanilla PP-GNN training with our optimized pipeline.  
   - Evaluate performance under different data placements:
     - **GPU memory**
     - **Host memory** (using **SGD-RR** or **SGD-CR**)
     - **Storage**

3. **Multi-GPU Experiments**  
   - Evaluate training with data in GPU and host memory using **SGD-RR** and **SGD-CR**.  
   - **Note:** Multi-GPU training does **not** support standard **SGD**.

4. **Automated Training Configuration Experiments**  
   - Run our automated system for optimizing training configurations.

---

## **Customization**  

New PP-GNN models can be added to the `gnn.py` file.  
To modify the training setup:  

- **Update `model_cfg.json`**  
  - Change the `"method"` parameter to **SIGN** or **SGC** to explore different models.  
  - Modify `"training_hops"` to experiment with different numbers of hops (the training_hops argument must not exceed the value of the same argument used for the preprocessing step, otherwise you need to redo the preprocessing with a larger training_hops). 

- **Modify `evaluation.sh`**  
  - Adjust **GPU IDs** for the main_mp.py and the **GPUcap** parameter for auto_run.py to use different numbers of GPUs.


