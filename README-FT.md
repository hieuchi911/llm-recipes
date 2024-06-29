# LLM-Recipes (Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs - Paper)

# Environment set up
- Run the [environment.yml](environment.yml) file to create a new conda environment and install the required packages:
  ```
  conda env create -f environment.yml
  conda activate venv
  ```

- Add an environment variable `PROJECT_DIR` with the value of the path from `$HOME` to `llm-distillation`. For example, for the case where the full path to `llm-distillation` is `/home1/hieutn/cs566/llm-distillation`, where `$HOME` is `/home1/hieutn`:
  ```
  export PROJECT_DIR=cs566
  ```

## Run Distillation Process
Below are some examples commands to run distillations in different hardware settings:

### Single node - multi-processes:
- Distill `google/gemma-7b-it` with `meta-llama/Meta-Llama-3-70B-Instruct` as the teacher:
    ```
    torchrun --standalone --nproc_per_node=2 \
    \
    finetuning.py \
    --model_name google/gemma-7b-it \
    --enable_fsdp \
    --lr 1e-6 \
    --num_epochs 5 \
    --dataset.file newssum.py \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --output_dir train/output/path \
    --save_step 100 \
    \
    --distillation \
    --distillation_config.model_name meta-llama/Meta-Llama-3-70B-Instruct \
    --distillation_config.pure_bf16 \
    --distillation_config.auto_dispatch \
    --distillation_config.distil_factor 1.5
    ```
- What does hardware specification look like for this?
    - execution is done on a single node (`--standalone`);
    - the student model is parallelized with FSDP across 2 GPU devices (`--nproc_per_node=2` and `--enable_fsdp`);
    - on each device there will be 1 the teacher model, both of which are dispatched across all available GPU devices in the node (`--distillation_config.auto_dispatch`)

### Multi-nodes - multi-processes:
(to be continued)

Some parameters used:
- `--model_name`: The ID of the student model (HuggingFace repository ID).
- `--lr`: Learning rate for the training process.
- `--num_epochs`: Number of epochs for training.
- `--batch_size_training`: Batch size for training.
- `--val_batch_size`: Batch size for validation.
- `--dataset.file`: Path to the loader script of the dataset to distill on.
- `--output_dir`: Directory to save the output.

- `--distillation`: Activate distillation.
- `--distillation_config.model_name`: The ID of the teacher model (HuggingFace repository ID).
- `--distillation_config.enable_fsdp`: Enable Fully Sharded Data Parallelism (FSDP).
- `--distillation_config.pure_bf16`: Use pure BF16 precision.
- `--distillation_config.distil_factor`: Factor for distillation loss.
- `--save_step`: Interval for saving checkpoints during training.
- `--encoder_decoder`: Specify this parameter if the student model follows an encoder-decoder architecture.
