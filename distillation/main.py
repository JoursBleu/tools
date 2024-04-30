import json
import torch
import transformers

from dataclasses import dataclass, field
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Trainer, GPTQConfig
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict, Optional, List

global local_rank

def rank0_print(*args):
    if local_rank == 0:
        print(*args)
rank0_print=print

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


training_args = transformers.TrainingArguments(output_dir="/mnt/volumes/cloudmodel-muses/lt/models/finetuned")
training_args.bf16 = True
training_args.num_train_epochs = 5
training_args.per_device_train_batch_size = 32
training_args.evaluation_strategy = "no"
training_args.save_strategy = "steps"
training_args.save_steps = 1000
training_args.save_total_limit = 4
training_args.learning_rate = 1e-5
training_args.weight_decay = 0.1
training_args.adam_beta2 = 0.95
training_args.warmup_ratio = 0.01
training_args.lr_scheduler_type = "cosine"
training_args.logging_steps = 10
training_args.model_max_length = 2048
training_args.gradient_checkpointing = False
training_args.deepspeed = "zero2.json"

model = transformers.AutoModelForCausalLM.from_pretrained(
    "/mnt/volumes/cloudmodel-muses/lt/models/Qwen1.5-MoE-A2.7B-Chat-small",
    cache_dir="/mnt/volumes/cloudmodel-muses/lt/lpex-cache",
    device_map="cuda",
    trust_remote_code=True,
    quantization_config=None,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    cache_dir="/mnt/volumes/cloudmodel-muses/lt/lpex-cache",
    use_fast=False,
    trust_remote_code=True,
)

tokenizer.pad_token_id = tokenizer.eod_id


data_args = DataArguments()
data_args.data_path = "/mnt/volumes/cloudmodel-muses/lt/data/qwen_data/drivelm_zh_valid.json"

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("json", data_files="/mnt/volumes/cloudmodel-muses/lt/data/qwen_data/drivelm_zh_valid.json")

data_module = make_supervised_data_module(
    tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
)

# Start trainner
trainer = Trainer(
    model=model, tokenizer=tokenizer, args=training_args, **data_module
)

trainer.train()
trainer.save_state()














