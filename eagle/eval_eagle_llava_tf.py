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

from llava.eval.llava_mixtral_eval import load_pretrained_model
from llava.eval.llava_mixtral_eval import create_data_loader

from eagle.model.cnets import Model
from eagle.model.configs import EConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--small_model_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-small")
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--use_sp', action="store_true", default=False)
    parser.add_argument('--draft_len', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=100)
    parser.add_argument('--benchmark_dataset_json', type=str, default="/lpai/volumes/cloudmodel-muses/llava_data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
    parser.add_argument('--image_dir', type=str, default="/lpai/volumes/cloudmodel-muses/llava_data/LLaVA-Pretrain/images")
    # "/lpai/volumes/cloudmodel-muses/lt/data/llava_data"

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

config = EConfig.from_pretrained(args.small_model_dir)
ea_layer = Model(config, load_checkpoint=args.small_model_dir+"/pytorch_model.bin", bias=False, total_tokens=args.draft_len, depth=args.draft_len, top_k=1, training=False).half().cuda(0).eval()

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

with open(args.benchmark_dataset_json, 'r') as f:
    questions = json.load(f)

questions = questions[40000:]

print("loading preprocess")
tokenizer, bigmodel, image_processor, context_len = load_pretrained_model(
            args.base_model_dir, model_base="llava_mixtral", device_map="auto", load_in_8bit=True, load_in_4bit=False)
dataset = create_data_loader(
    questions,
    args.image_dir,
    tokenizer,
    image_processor,
    bigmodel.config,
    batch_size=args.batch_size
)
small_head = torch.nn.Linear(in_features=7168, out_features=64000, bias=False)
small_head_weight = bigmodel.lm_head.weight.data
small_head.weight.data = small_head_weight.half().cuda(0)

total_time = 0
total_token = 0
avg_acc_lens = 0

count = 0
avg_acc_len = 0
total_input_len = 0
total_base_time = 0.
total_vit_time = 0.


truth = torch.tensor([[647,  3807,   651,   562,  3589, 23956,   597,   562,  5083,
         23956, 15979,    98, 59568]])

with torch.inference_mode():
    # while count < args.benchmark_steps:
    for idx, data in enumerate(dataset):
        ea_layer.reset_kv()
        ea_layer.init_tree()
        (input_ids, image_tensor, image_sizes, prompt) = data
        print("prompt:", prompt)
        # print("input_ids:", input_ids)

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # breakpoint()
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

        # breakpoint()
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

        input_token_len = input_ids.shape[1]
        # input_token_len = input_embedding.shape[1]

        print("input_ids", input_ids)
        print("position_ids", position_ids)
        print("attention_mask", attention_mask)
        print("past_key_values", past_key_values)
        print("input_embedding", input_embedding.shape)
        print("labels", labels)

        # breakpoint()
        print("Generating...")
        start = time.time()
        status = 0
        steps = 0
        new_tokens_num = 0
        # breakpoint()
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
        big_new_tokens = torch.argmax(out["logits"][:, -1:], dim=-1).cuda(0)
        input_ids = torch.concat((input_ids.cuda(0), big_new_tokens), -1)
        new_embedding = bigmodel.model.embed_tokens(big_new_tokens)
        input_embedding = torch.concat((input_embedding, new_embedding), 1)
        hidden_states = out.hidden_states[-1]
        new_tokens_num += 1
        steps += 1
        total_accept_len = 0
        # breakpoint()
        ea_out = ea_layer.topOne_genrate(
            hidden_states=hidden_states.cuda(0),
            # input_ids=big_new_tokens,
            inputs_embeds=input_embedding,
            head=small_head,
            logits_processor=None
        )
        # phrase_tokens, new_token_len, status = choices.get_phrase_token(input_ids)
        # print("new_tokens_num", new_tokens_num)
        # print("phrase_tokens", phrase_tokens)

        while (new_tokens_num < args.max_new_tokens):
            # breakpoint()
            # input_ids = bigmodel.prepare_inputs_for_generation(
                # input_ids=torch.concat((input_ids, input_ids), dim=1),
                # past_key_values=out["past_key_values"],
                # use_cache=True,
            # )
            if args.use_sp:
                draft_new_tokens = torch.concat((big_new_tokens[:,-1:], ea_out[0]), dim=1)
            else:
                draft_new_tokens = big_new_tokens
            # draft_new_tokens = truth.cuda(0)
            # breakpoint()
            out = bigmodel(
                input_ids=draft_new_tokens,
                past_key_values=out["past_key_values"],
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = out.hidden_states[-1]
            big_new_tokens = torch.argmax(out["logits"], dim=-1)
            posterior_mask = (draft_new_tokens[:, 1:] == big_new_tokens[:, :-1]).int()
            accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
            accept_length += 1
            total_accept_len += accept_length - 1
            if False:
                print("input_ids[:, :]", input_ids)
                print("draft_new_tokens[:, :]", draft_new_tokens)
                print("draft_new:", tokenizer.decode(draft_new_tokens[0]))
                print("big_new_tokens[:, :]", big_new_tokens)
                print("big_new:", tokenizer.decode(big_new_tokens[0]))
            big_new_tokens = big_new_tokens[:, :accept_length]
            new_embedding = bigmodel.model.embed_tokens(big_new_tokens)
            input_embedding = torch.concat((input_embedding, new_embedding), 1)
            seq_len = input_embedding.shape[1] - 1
            input_ids = torch.concat((input_ids, big_new_tokens), -1)
            # phrase_tokens, new_token_len, status = choices.get_phrase_token(input_ids)
            past_key_values = []
            for i in range(len(out["past_key_values"])):
                past_key_values.append([out["past_key_values"][i][0][:,:,:seq_len], out["past_key_values"][i][1][:,:,:seq_len]])
            out["past_key_values"] = past_key_values
            new_tokens_num += accept_length
            steps += 1
            # breakpoint()
            # print("new_tokens_num", new_tokens_num)
            # print("phrase_tokens", phrase_tokens)
            if (big_new_tokens[0, -1] == tokenizer.eos_token_id):
                break
            # breakpoint()
            hidden_states = hidden_states[:, :accept_length]
            ea_out = ea_layer.topOne_genrate(
                hidden_states=hidden_states.cuda(0),
                # input_ids=big_new_tokens,
                inputs_embeds=input_embedding,
                head=small_head,
                logits_processor=None
            )

        end = time.time()

        # print(input_ids)
        new_token = new_tokens_num
        # generation_logits = input_ids['generation_logits']
        # print("outputs_id:", input_ids.cpu())
        avg_acc_len = total_accept_len * 1. / steps
        if True:
            input_ids = input_ids[:, input_token_len:]
        # print("generation_logits", generation_logits.shape)
        print("Step", idx, ":", end - start, "s")
        print("Outputs:", tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        print("input_ids.numel():", input_ids.numel())
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
            print("avg_acc_lens:", avg_acc_lens / (idx-1))
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

