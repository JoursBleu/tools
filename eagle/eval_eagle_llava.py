import argparse
import json
import torch
import time

from datasets import load_dataset
from eagle.model.ea_model import EaModel
from eagle.model.modeling_llama_kv import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HfLlamaForCausalLM
from mind_ad.model.builder import load_pretrained_model
from mind_ad.eval.eval_llava import create_data_loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--small_model_dir', type=str, default=None)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--use_sp', action="store_true", default=False)
    parser.add_argument('--benchmark', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=10)
    parser.add_argument('--benchmark_dataset', type=str, default="ccdv/cnn_dailymail")
    parser.add_argument('--benchmark_dataset_version', type=str, default="3.0.0")
    parser.add_argument('--benchmark_dataset_json', type=str, default=None)

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
print("loading model")

model = EaModel.from_pretrained(
    base_model_path=args.base_model_dir,
    ea_model_path=args.small_model_dir,
    torch_dtype=torch.float16,
    load_in_4bit=False,
).eval().cuda()

# print("compiling...")
# model.draft_one=torch.compile(model.draft_one, mode="reduce-overhead", fullgraph=True,dynamic=True)
# model.base_forward=torch.compile(model.base_forward, mode="reduce-overhead", fullgraph=True,dynamic=True)
# model.base_forward_one=torch.compile(model.base_forward_one, mode="reduce-overhead", fullgraph=True)
# print("done compile.")

with open(args.benchmark_dataset_json, 'r') as f:
    questions = json.load(f)
    # questions = questions[57000:]

tokenizer, bigmodel, image_processor, context_len = load_pretrained_model(
            args.base_model_dir, model_base="llava_qwen", device_map="cuda", load_in_8bit=False, load_in_4bit=False)
dataset = create_data_loader(
    questions,
    "/mnt/volumes/cloudmodel-muses/llava_data",
    tokenizer,
    image_processor,
    bigmodel.config,
    conv_mode="qwen_instruct",
    padding_side="right",
    batch_size=args.batch_size
)

# model.base_model = bigmodel


total_time = 0
total_token = 0
avg_acc_lens = 0

count = 0
avg_acc_len = 0
total_input_len = 0
total_base_time = 0.

# while count < args.benchmark_steps:
for idx, data in enumerate(dataset):
    (input_ids, image_tensor, image_sizes, prompt) = data
    print("prompt:", prompt)
    print("input_ids:", input_ids)

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    input_len = input_ids.shape[1]

    with torch.inference_mode():
        position_ids=None
        attention_mask=None
        inputs_embeds=None
        (
            inputs,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
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

        base_start = time.time()
        # output_ids = bigmodel.generate(
            # position_ids=position_ids,
            # attention_mask=attention_mask,
            # inputs_embeds=inputs_embeds
        # )
        base_end = time.time()
        # print("Outputs:", tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    start = time.time()
    if (args.use_sp):
        outputs, avg_acc_len = model.eagenerate(input_ids.cuda(), inputs_embeds.cuda(), max_new_tokens=args.max_new_tokens)
        # outputs, avg_acc_len = model.eagenerate(batch.cuda(), top_p=0.9, temperature=0.5, max_new_tokens=args.max_new_tokens)
    else:
        outputs = model.naivegenerate(input_ids.cuda(), inputs_embeds.cuda(), max_new_tokens=args.max_new_tokens)
        # outputs = model.naivegenerate(batch.cuda(), top_p=0.9, temperature=0.5, max_new_tokens=args.max_new_tokens)
    end = time.time()
    print("outputs_id:", outputs.cpu())
    outputs = outputs[:, input_len:]
    print("Step", idx, ":", end - start, "s")
    print("Outputs:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    new_token = outputs.numel()
    print("Total new token:", new_token)
    print("Avg acc len:", avg_acc_len)

    if (idx > 1):
        count += 1
        total_time += (end - start)
        total_token += new_token
        avg_acc_lens += avg_acc_len
        total_input_len += input_len
        total_base_time += (base_end - base_start)
        print("avg_acc_lens:", avg_acc_lens / (idx+1))
    if (idx == args.benchmark_steps):
        break
print("Total steps:", args.benchmark_steps)
print("Batch size:", args.batch_size)
print("Input len:", total_input_len / count)
print("Avg output len:", total_token / count)
print("Avg acc len:", avg_acc_lens / count)
print("benchmark_steps:", count)
print("BASE TPS:", total_token / total_base_time)
print("TPS:", total_token / total_time)

