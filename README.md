# SwinLLama: Enhancing Medical report generation with Advanced Large Language Models


<!-- ## Introduction
![overview](figs/overview.png) -->

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
git clone https://github.com/wang-zhanyu/SwinLLama.git
cd SwinLLama
pip install -r requirements.txt
```


<!-- **2. Prepare the pretrained Vicuna weights**

The current version of SwinLLama is built on the Vicuna-7B.
Please use the fiollowing command to prepare the Vicuna weights.

```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command to create the final working weight

```
cd SwinLLama
python -m fastchat.model.apply_delta --base decapoda-research/llama-7b-hf  --target vicuna_weights --delta lmsys/vicuna-7b-delta-v0
```

The final weights would be in a single folder with the following structure:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[configs/config_mimic.py](configs/config_mimic.py#L28) at Line 28. -->

<!-- 
**3. Prepare the pretrained SwinLlama checkpoint**

Download our pretrained checkpoint from
[coming soon].
Then, set the path to the delta_file in the config file 
in [configs/config_mimic.py](configs/config_mimic.py#L18) at Line 18.  -->


**2. Prepare the training dataset**

We train SwinLLama on the MIMIC-CXR dataset
You can dowmload the dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and set base_dir to the file download path in the config file in [configs/config_mimic.py](configs/config_mimic.py#L8) at Line 8, for example, base_dir='physionet.org/files/mimic-cxr-jpg/2.0.0/files'.

The pre-processed reports can be download from [here](https://drive.google.com/file/d/16NvBAaiAEgBacW4CGDd5lI8OsR-WZeDz/view?usp=sharing), after dwonlading, set the path of this file in the config file in [configs/config_mimic.py](configs/config_mimic.py#L7) at Line 7.


### Launching Demo Locally

Try out our demo [demo.py](demo.py) on your local machine by running

```
python demo.py --delta_file /path/to/pretrained/checkpoint
```

### Training

To launch the training, run the following command. In our experiments, we use 2 3090GPU. 
You can change the save path in the config file 
[configs/config_mimic.py](configs/config_mimic.py#L16)

```bash
python train.py --gpus 2 --batch_size 4 --val_batch_size 4 --max_epochs 3 --savedmodel_path /path/to/savemodel
```

## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Vicuna](https://github.com/lm-sys/FastChat) The fantastic language ability of Vicuna with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
