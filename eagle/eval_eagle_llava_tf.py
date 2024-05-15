import argparse
import json
import math
import numpy as np
import time
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Optional, Tuple, Union

import mind_ad
from mind_ad.model.builder import load_pretrained_model
from mind_ad.eval.eval_llava import create_data_loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--small_model_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-small")
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--use_sp', action="store_true", default=False)
    parser.add_argument('--benchmark', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=10)
    parser.add_argument('--benchmark_dataset_json', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default="/mnt/volumes/cloudmodel-muses/llava_data")
    # "/lpai/volumes/cloudmodel-muses/lt/data/llava_data"

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

with open(args.benchmark_dataset_json, 'r') as f:
    questions = json.load(f)

print("loading preprocess")
tokenizer, bigmodel, image_processor, context_len = load_pretrained_model(
            args.base_model_dir, model_base="llava_qwen", device_map="cuda", load_in_8bit=False, load_in_4bit=False)
dataset = create_data_loader(
    questions,
    args.image_dir,
    tokenizer,
    image_processor,
    bigmodel.config,
    conv_mode="qwen_instruct",
    padding_side="right",
    batch_size=args.batch_size
)
llm_model_config = big_runner.session._model_config
llm_model_config.tp_size = 1
llm_table_size = llm_model_config.vocab_size

# keys = {
    # "场景": 1,
    # "时间",
    # "道路类型",
    # "备选车道",
    # "自车车道",
    # "文本",
# }
phrases_list = [{
    42192: [42192, 100183, 45995, pad_id], # "无灯路"
    104527: [104527, 114913, 45995, pad_id], # "坑洼路"
    100846: [100846, 119340, 45995, pad_id], # "泥泞路"
    100817: [100817, pad_id, pad_id, pad_id], # "事故"
    101178: [101178, 100686, pad_id, pad_id], # "道路施工"
    104875: [104875, 99425, 99679, 100183], # "临时红绿灯"
}]
drafts_list = [{
    "middle": [11],
    "end": [
        [20450, 25, 42192, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # 时间:无;道路类型:无;备选车道:[
        [20450, 25, 106772, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # 时间:白天;道路类型:无;备选车道:[
        [20450, 25, 108036, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # 时间:夜晚;道路类型:无;备选车道:[
    ],
}]
status_end_list = {
    0: 5265,
}

for phrases in phrases_list:
    for key in phrases:
        phrases[key] = (torch.Tensor(phrases[key]).to(torch.int32).unsqueeze(0).cuda(), len(phrases[key]))

def get_phrase_token(new_token, current_status=None):
    new_token_len = 1
    if current_status is not None and current_status < len(phrases_list):
        new_token_item = new_token[0, -1].item()
        # print("current_status", current_status)
        # print("new_token_item", new_token_item)
        if status_end_list[current_status] == new_token_item:
            current_status += 1
        else:
            phrases = phrases_list[current_status]
            if new_token_item in phrases:
                new_token, new_token_len = phrases[new_token_item]
                # print("prase", new_token)
    if new_token_len ==  1:
        # print("base new_token:", new_token)
        new_token = torch.concat((new_token, torch.tensor([[pad_id, ]]).to(torch.int32).cuda()), dim=-1)
        new_token_len = 2
        # print("fixed new_token:", new_token)

    return new_token, new_token_len, current_status

def get_prefix():
    prefix = torch.Tensor([102122, 7259]).to(torch.int32).unsqueeze(0).cuda()
    return prefix


total_time = 0
total_token = 0
avg_acc_lens = 0

count = 0
avg_acc_len = 0
total_input_len = 0
total_base_time = 0.
total_vit_time = 0.


with torch.inference_mode():
    # while count < args.benchmark_steps:
    for idx, data in enumerate(dataset):
        llm_table_size = llm_model_config.vocab_size
        (input_ids, image_tensor, image_sizes, prompt) = data
        print("prompt:", prompt)
        # print("input_ids:", input_ids)

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        vit_start = time.time()
        position_ids=None
        attention_mask=None
        input_embedding=None
        (
            inputs,
            position_ids,
            attention_mask,
            _,
            input_embedding,
            _
        ) = bigmodel.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes
        )
        vit_end = time.time()

        base_start = time.time()
        # output_ids = bigmodel.generate(
            # position_ids=position_ids,
            # attention_mask=attention_mask,
            # input_embedding=input_embedding
        # )
        base_end = time.time()
        # print("Outputs:", tokenizer.batch_decode(output_ids, skip_special_tokens=True))

        # input_token_len = input_ids.shape[1]
        input_token_len = input_embedding.shape[1]

        start = time.time()
        input_ids = torch.concat((input_ids, get_prefix()), dim=-1)

        bigmodel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        end = time.time()

        # print(outputs)
        new_token = outputs['sequence_lengths'] - input_token_len + 2
        # generation_logits = outputs['generation_logits']
        outputs = outputs['output_ids']
        # print("outputs_id:", outputs.cpu())
        if False:
            outputs = outputs[:, input_token_len:]
        # print("generation_logits", generation_logits.shape)
        print("Step", idx, ":", end - start, "s")
        print("Outputs:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print("outputs.numel():", outputs.numel())
        print("Total new token:", new_token)
        print("Avg acc len:", avg_acc_len)

        if (idx > 1):
            count += 1
            total_time += (end - start)
            total_token += new_token
            avg_acc_lens += avg_acc_len
            total_input_len += input_token_len
            total_base_time += (base_end - base_start)
            total_vit_time += (vit_end - vit_start)
            print("avg_acc_lens:", avg_acc_lens / (idx+1))
        if (idx == args.benchmark_steps):
            break
print("Total steps:", args.benchmark_steps)
print("Batch size:", args.batch_size)
print("Input len:", total_input_len / count)
print("Avg output len:", total_token / count)
print("Avg acc len:", avg_acc_lens / count)
print("Avg tokens / step:", avg_acc_lens / count + 1)
print("Avg vit time:", total_vit_time / count)
print("Avg total time:", (total_vit_time + total_time) / count)
print("benchmark_steps:", count)
print("BASE TPS:", total_token / total_base_time)
print("TPS:", total_token / total_time)

