import argparse
import ast


parser = argparse.ArgumentParser(description="hyper-parameter for SwinLLama")
# ========================= Dataset Configs ==========================
parser.add_argument('--annotation', type=str, default=r'data/my_mimic_anno.json', help="mimic annotation")
parser.add_argument('--base_dir', type=str, default=r'/home/zhanyu_wang/data/mimic_cxr/images')
parser.add_argument('--batch_size', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--val_batch_size', default=4, type=int, help="use for validation duration per worker")
parser.add_argument('--test_batch_size', default=4, type=int, help="use for testing duration per worker")
parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--cpu_num', default=4, type=int, help="Cpu num for dataloaders")

# ======================== SavedModel Configs =========================
parser.add_argument('--savedmodel_path', type=str, default='save/mimic/swinllama_v1')
parser.add_argument('--ckpt_file', type=str, default=None)
parser.add_argument('--delta_file', type=str, default=None)

# ========================= Learning Configs ==========================
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')

# ========================= LLama R2Gen ==========================
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--vit_precision', default='fp16', type=str)
parser.add_argument('--freeze_swin', default=False, type=bool)
parser.add_argument('--use_lora', default=True, type=bool)
parser.add_argument('--llama_model', default='eachadea/vicuna-7b-1.1', type=str)
parser.add_argument('--prompt_template', default='###Human: {} ###Radiologist: ', type=str)
parser.add_argument('--max_txt_len', default=120, type=int)
parser.add_argument('--low_resource', default=False, type=bool)
parser.add_argument('--end_sym', default='\n', type=str)
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--num_beam_groups', type=int, default=1)
parser.add_argument('--report_min_length', type=int, default=100)
parser.add_argument('--report_max_length', type=int, default=120)
parser.add_argument('--repetition_penalty', type=float, default=1)
parser.add_argument('--length_penalty', type=float, default=1.0)
parser.add_argument('--diversity_penalty', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--weights', type=list, default=[0.5, 0.5])
parser.add_argument('--scorer_types', type=list, default=['Bleu_4', 'CIDEr'])

# ====================== Pytorch Lightning ===========================
parser.add_argument('--gpus', type=int, default=1, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='how many machines to use')
parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator types')
parser.add_argument('--strategy', type=str, default="ddp", help='default ddp for multi-gpus')
parser.add_argument('--amp_backend', type=str, default="native", help='The mixed precision backend to use ("native" or "apex")')
parser.add_argument('--precision', type=int, default=16, help='16 or 32, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How many steps runs when validation')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How many steps runs when training')
parser.add_argument('--max_steps', default=1500000, type=int, metavar='N', help='number of total step to run')
parser.add_argument('--max_epochs', type=int, default=3, help='number of total epoches to run')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=0.3, help='How often to check the validation set')
parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')
