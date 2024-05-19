import argparse
import json
import math
import numpy as np
import time
import torch

from datasets import load_dataset
from enum import IntEnum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import StaticCache
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

phrases_lists = [
# start
{
    # 53599:  [53599, 118, 85254, 25], # ' 场景:'
    53599:  [53599, 118, 85254, 7259], # ' 场景:['
},
# changjing
{
    # [];时间:'
    25:     [25, 1294, 26, 20450, 25],
    # '无灯路];时间:'
    42192:  [42192, 100183, 45995, 5265, 20450, 25, ], 
    # '坑洼路];时间:'
    104527: [104527, 114913, 45995, 5265, 20450, 25, ],
    # '泥泞路];时间:'
    100846: [100846, 119340, 45995, 5265, 20450, 25, ],
    # '事故];时间:'
    100817: [100817, 5265, 20450, 25, ], 
    # '道路施工];时间:'
    101178: [101178, 100686, 5265, 20450, 25, ],
    # '临时红绿灯];时间:'
    104875: [104875, 99425, 99679, 100183, 5265, 20450, 25, ],
},
# shijian 
{
    42192:  [42192, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # '无;道路类型:无;备选车道:['
    106772: [106772, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # '白天;道路类型:无;备选车道:['
    108036: [108036, 26, 101178, 31905, 25, 42192, 26, 56278, 30767, 111363, 7259], # '夜晚;道路类型:无;备选车道:['
},
# doulu
{},
# beixuan
{
    42192:  [42192, 11,], # '无,'
    100714: [100714, 11,], # '普通,'
    5265:   [5265, 35926, 39953, 111363, 25, 42192, 26, 108704, 25], # '];自车车道:无;文本:'
},
# ziche
{
    42192: [42192, 26, 108704, 25, 103978], # '无;文本:车辆'
    111687: [111687, 26, 108704, 25, 103978], # '左侧;文本:车辆'
    113658: [113658, 26, 108704, 25, 103978], # '右侧;文本:车辆'
    104399: [104399, 26, 108704, 25, 103978], # '右侧;文本:车辆'
    103583: [103583, 44793, 26, 108704, 25, 103978], # '单车道;文本:车辆'
},
# wenben
{
    # '所在左侧车道允许行驶,也可选择右侧相邻车道;'
    101393: [101393, 111687, 111363, 102496, 106019, 11, 106332, 50404, 113658, 118616, 111363, 26],
    # '位于中间车道,左右均有邻车道可供选择;'
    103987: [103987, 104399, 111363, 11, 101081, 108549, 100603, 111363, 111279, 50404, 26],
    101081: [101081, 108549, 100603, 111363, 111279, 50404, 26], # '左右均有邻车道可供选择;'
    # '驶在左侧车道,'
    100105: [100105, 18493, 111687, 111363, 11],
    # '可以选择保持或移至右侧相邻车道;'
    108325: [108325, 100662, 57191, 59534, 56137, 113658, 118616, 111363, 26],
    102972: [102972, 111363, 15946, 26], # '不在车道中;'
    100004: [100004, 102972, 99885, 111363, 17447, 26], # '目前不在任何车道上;'
    105307: [105307, 102779, 100692, 9370, 111363, 26], # '尚未占据明确的车道;'
    38342:  [38342, 101199, 113743, 9370, 111363, 15946, 26], # '未处于标记的车道中;'
    99885:  [99885, 111363, 15946, 26], # '任何车道中;'
    65676:  [65676, 105327, 111363, 101065, 26], # '非正规车道区域;'
    67949:  [67949, 38342, 18493, 99885, 111363, 17447, 26], # '当前未在任何车道上;'
    # '前方施工,车辆将慢速行驶,避免急刹车. '
    108348: [108348, 100686, 11, 103978, 44063, 99843, 94299, 106019, 11, 101153, 99508, 110501, 13, 220],
    # [前方施工]'情况,车辆将慢速前进,避免急刹. '
    99559:  [99559, 11, 103978, 44063, 99843, 94299, 105883, 11, 101153, 99508, 101796, 13, 220],
    100662: [100662, 67949, 111363, 106019], # '持当前车道行驶'
    # '道路光线不足,车辆将选择慢速前进,'
    101178: [101178, 109587, 102004, 11, 103978, 44063, 50404, 99843, 94299, 105883, 11],
    # '车辆将选择慢速前进'
    103978: [103978, 44063, 50404, 99843, 94299, 105883, 11,],
    # '避免突发事件导致的急刹. '
    101153: [101153, 116278, 100673, 9370, 99508, 101796, 13, 220],
    104875: [104875, 99735, 104757, 100183], # '临时交通信号灯'
    104875: [104875, 99425, 99679, 100183], # '临时红绿灯'
    # '路口中心,安装了一个临时交通信号灯. '
    108005: [108005, 99488, 11, 103999, 104059, 104875, 99735, 104757, 100183, 13, 220],
    103999: [103999, 104059, 104875, 99735, 104757, 100183, 13, 220], # '安装了一个临时交通信号灯. '
},
]
for phrases_list in phrases_lists:
    for key in phrases_list:
        phrases_list[key] = (torch.Tensor(phrases_list[key]).to(torch.int32).unsqueeze(0).cuda(), len(phrases_list[key]))

class Status(IntEnum):
    start = 0
    changjing = 1
    shijian = 2
    doulu = 3
    beixuan = 4
    ziche = 5
    wenben = 6

class Choices:

    def __init__(self):
        self.status = Status.start
        self.is_empty_changjing = False
        self.is_temp_light = False
        self.debug = False

    def get_phrase_token(self, output_ids):
        new_token_len = 1
        # print("phrases", phrases_list)
        new_token = output_ids[:, -1:]
        new_token_item = new_token.item()
        out_str = tokenizer.decode(output_ids[0])
        if self.debug:
            print("string:", out_str)
        if "文本" in out_str:
            self.status = Status.wenben
        elif "自车车道" in out_str:
            self.status = Status.ziche
        elif "备选车道" in out_str:
            self.status = Status.beixuan
        # elif "道路类型" in out_str:
            # status = Status.doulu
        elif "时间" in out_str:
            self.status = Status.shijian
        elif "场景" in out_str:
            self.status = Status.changjing
        else:
            self.status = Status.start

        phrases_list = phrases_lists[int(self.status)]
        if self.debug:
            print("new_token_item", new_token_item)
        if new_token_item in phrases_list:
            new_token, new_token_len = phrases_list[new_token_item]
            # new_tokens = torch.concat((output_ids[:, -1:], new_token), -1)
            if self.debug:
                print("猜:", tokenizer.decode(new_token[0]))

        if self.debug:
            print("self.status", self.status)
            print("new_tokens", new_token)
            print("new_token_len", new_token_len)
        return new_token, new_token_len, None


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
            past_key_values,
            input_embedding,
            labels
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

        print("Compiling...")
        # bigmodel = torch.compile(bigmodel, backend="inductor")

        # input_token_len = input_ids.shape[1]
        input_token_len = input_embedding.shape[1]

        print("inputs", inputs)
        print("position_ids", position_ids)
        print("attention_mask", attention_mask)
        print("past_key_values", past_key_values)
        print("input_embedding", input_embedding.shape)
        print("labels", labels)

        print("Generating...")
        start = time.time()
        choices = Choices()
        status = 0
        steps = 0
        new_tokens_num = 0
        past_key_values = StaticCache(bigmodel.config, max_batch_size=4, max_cache_len=1024, device=torch.device("cuda"))
        out = bigmodel(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embedding,
            labels=labels,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True
        )
        # print("out['logits']", out['logits'].shape)
        outputs = torch.argmax(out["logits"][:, -1:], dim=-1)
        new_tokens_num += 1
        steps += 1
        total_accept_len = 0
        phrase_tokens, new_token_len, status = choices.get_phrase_token(outputs)
        # print("new_tokens_num", new_tokens_num)
        # print("phrase_tokens", phrase_tokens)

        while (new_tokens_num < args.max_new_tokens):
            out["past_key_values"].set_seen_tokens(input_token_len+new_tokens_num)
            inputs = bigmodel.prepare_inputs_for_generation(
                input_ids=phrase_tokens,
                past_key_values=out["past_key_values"],
                use_cache=True,
            )
            out = bigmodel(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"],
                past_key_values=inputs["past_key_values"],
                labels=labels,
                use_cache=True,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=True,
            )
            big_new_tokens = torch.argmax(out["logits"], dim=-1)
            if new_token_len > 1:
                posterior_mask = (phrase_tokens[:, 1:] == big_new_tokens[:, :-1]).int()
                accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
                accept_length += 1
            else:
                accept_length = torch.Tensor([1]).to(torch.int32).cuda()
            total_accept_len += accept_length - 1
            big_new_tokens = big_new_tokens[:, :accept_length]
            outputs = torch.concat((outputs, big_new_tokens), -1)
            phrase_tokens, new_token_len, status = choices.get_phrase_token(outputs)
            new_tokens_num += new_token_len
            steps += 1
            # print("new_tokens_num", new_tokens_num)
            # print("phrase_tokens", phrase_tokens)
            if (big_new_tokens[0, -1] == tokenizer.eos_token_id):
                break

        end = time.time()

        # print(outputs)
        new_token = new_tokens_num
        # generation_logits = outputs['generation_logits']
        # print("outputs_id:", outputs.cpu())
        avg_acc_len = total_accept_len * 1. / steps
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

