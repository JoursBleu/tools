import argparse
import json
import math
import numpy as np
import sys
import time
import torch

from datasets import load_dataset
from eagle.model.ea_model import EaModel
from enum import IntEnum
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Optional, Tuple, Union

import mind_ad
from mind_ad.model.builder import load_pretrained_model
from mind_ad.eval.eval_llava import create_data_loader

import tensorrt_llm
from tensorrt_llm import ModelConfig
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp
from tensorrt_llm.runtime.generation  import SamplingConfig, StoppingCriteria, LogitsProcessor, RuntimeTensor
from tensorrt_llm.runtime.kv_cache_manager import GenerationSequence, KVCacheManager, KVCacheUpdater

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--eagle_small_model_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4/")
    parser.add_argument('--small_model_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4/")
    parser.add_argument('--big_engine_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt")
    parser.add_argument('--small_engine_dir', type=str, default=None)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--use_small_head', action="store_true", default=False)
    parser.add_argument('--mds_len', type=int, default=0)
    parser.add_argument('--use_table_search', action="store_true", default=False)
    parser.add_argument('--benchmark', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_steps', type=int, default=10)
    parser.add_argument('--benchmark_dataset_json', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default="/mnt/volumes/cloudmodel-muses/llava_data")
    # "/lpai/volumes/cloudmodel-muses/lt/data/llava_data"

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

args.use_mds = (args.mds_len > 0)

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

print("loading model")
draft_token_len = args.mds_len
medusa_choices = [[0] * draft_token_len] if args.use_mds else None

medusa_packed_mask = torch.zeros(
    (1, draft_token_len + 1, 1),
    dtype=torch.int32,
    device='cuda')
for i in range(draft_token_len + 1):
    medusa_packed_mask[0,i,0] = (2 ** (i+1)) - 1
medusa_position_offsets = torch.arange(draft_token_len+1, dtype=torch.int32, device='cuda').unsqueeze(0)

prompt_table_path = "/root/.cache/prompt_table.npy"
big_runner = ModelRunner.from_dir(
    engine_dir=args.big_engine_dir,
    rank=tensorrt_llm.mpi_rank(),
    medusa_choices=medusa_choices,
)
torch.cuda.set_stream(big_runner.session.stream)

eagle_model = EaModel.from_pretrained(
    base_model_path=args.base_model_dir,
    ea_model_path=args.eagle_small_model_dir,
    ea_engine_path=args.small_engine_dir,
    torch_dtype=torch.float16,
    load_in_4bit=False,
).eval().cuda()

with open(args.benchmark_dataset_json, 'r') as f:
    questions = json.load(f)

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

def ptuning_setup(embedding_table, input_ids):
    task_vocab_size = torch.tensor(
        [embedding_table.shape[1]],
        dtype=torch.int32,
    ).cuda()
    embedding_table = embedding_table.view(
        (embedding_table.shape[0] * embedding_table.shape[1],
         embedding_table.shape[2]))

    hidden_size = llm_model_config.hidden_size * llm_model_config.tp_size
    assert embedding_table.shape[
        1] == hidden_size, "Prompt table dimensions do not match hidden size"

    embedding_table = embedding_table.cuda().to(
        dtype=tensorrt_llm._utils.str_dtype_to_torch(
            llm_model_config.dtype))

    tasks = torch.zeros([args.batch_size], dtype=torch.int32).cuda()
    # tasks = torch.zeros([batch_size], dtype=torch.int32)

    # return [embedding_table, tasks, task_vocab_size]
    return {
        'prompt_embedding_table': embedding_table.cuda(),
        'tasks': tasks.cuda(),
        'prompt_vocab_size': task_vocab_size.cuda()
    }

def setup_fake_prompts(input_embedding):
    # Assemble fake prompts which points to input_embedding actually
    fake_prompt_id = torch.arange(
        llm_model_config.vocab_size, llm_model_config.vocab_size +
        input_embedding.shape[0] * input_embedding.shape[1])
    fake_prompt_id = fake_prompt_id.reshape(input_embedding.shape[0],
                                            input_embedding.shape[1])

    input_ids = fake_prompt_id.contiguous().to(torch.int32)

    ptuning_args = ptuning_setup(input_embedding, input_ids)

    # if tensorrt_llm.mpi_rank() == 0:
        # prompt_table = ptuning_args['prompt_embedding_table']
        # prompt_table = torch.stack([prompt_table])
        # np.save(prompt_table_path, torch_to_numpy(prompt_table))
    return input_ids, ptuning_args


# seen token map
seen_tokens_set = set([])
indexes = [
11, 16, 25, 26, 118, 220, 1294, 5265, 7259, 16872, 17447, 18600, 20450, 30709, 30767, 31905, 35568, 35727, 35926, 39953, 42192, 44636, 44793, 45995, 53599, 56278, 57191, 85254, 86119, 99630, 100029, 100167, 100183, 100205, 100519, 100686, 100714, 100806, 100817, 101142, 101178, 102540, 103583, 104307, 104399, 104527, 105212, 106772, 107213, 107500, 108005, 109131, 109259, 111363, 111687, 112018, 113048, 113658, 114411, 114913, 117507, 151643,
]
indexes = torch.Tensor(indexes).to(torch.int32).cuda()
print("eagle_model.base_model.lm_head", eagle_model.base_model.lm_head, flush=True)
slice_weight = eagle_model.base_model.lm_head.weight.data[indexes, :]
small_head = torch.nn.Linear(slice_weight.shape[1], slice_weight.shape[0], bias=False, device=slice_weight.device, dtype=slice_weight.dtype)
print("small_head", small_head, flush=True)
small_head.weight.data = slice_weight
eagle_model.ea_layer.indexes = indexes
eagle_model.ea_layer.small_head = small_head
eagle_model.ea_layer.use_small_head = args.use_small_head

fake_draft_tensor = torch.tensor([[pad_id,] * (draft_token_len + 1)]).to(torch.int32).cuda()
print("fake_draft_tensor", fake_draft_tensor)

phrases_lists = [
# changjing
{
    # ' 场景'
    53599: [53599, 118, 85254],
    # ':[];'
    25: [25, 1294, 26],
    # ':[公交车道];'
    7259: [7259, 107500, 44793, 5265,],
},
# tianqi
{
    # '天气:'
    104307: [104307, 25,],
    # '无;'
    42192: [42192, 26,],
    # '晴阴天;'
    105212: [105212, 100205, 35727, 26,],
},
# shijian 
{
    # '时间:'
    20450: [20450, 25,],
    # '无;'
    42192: [42192, 26,],
    # '白天;'
    106772: [106772, 26,],
    # '黑夜;'
    117507: [117507, 26,],
},
# daolu
{
    # '道路类型:'
    101178: [101178, 31905, 25,],
    # '普通道路;'
    100714: [100714, 101178, 26,],
    # '高速;'
    100806: [100806, 26,],
    # '无;'
    42192: [42192, 26,],
},
# beixuan
{
    # '备选车道:['
    56278: [56278, 30767, 111363, 7259,],
    # '无,'
    42192: [42192, 11,],
    # '普通,'
    100714: [100714, 11,],
    # '公交];'
    102540: [102540, 5265,],
},
# ziche
{
    # '自车车道:'
    35926: [35926, 39953, 111363, 25,],
    # '无; <eos>'
    42192: [42192, 26, 220, 151643,],
    # '右侧; <eos>'
    113658: [113658, 26, 220, 151643,],
    # '中间; <eos>'
    104399: [104399, 26, 220, 151643,],
    # '左侧; <eos>'
    111687: [111687, 26, 220, 151643,],
},
# wenben
{
},
]
for phrases_list in phrases_lists:
    for key in phrases_list:
        phrases_list[key] = (torch.Tensor(phrases_list[key]).to(torch.int32).unsqueeze(0).cuda(), len(phrases_list[key]))

prefix = torch.Tensor([53599, 118, 85254]).to(torch.int32).reshape([1, 3]).cuda()
prefix_len = 3

class Status(IntEnum):
    changjing = 0
    tianqi = 1
    shijian = 2
    daolu = 3
    beixuan = 4
    ziche = 5
    wenben = 6

class Choices:

    def __init__(self):
        # self.status = Status.start
        self.is_empty_changjing = False
        self.is_temp_light = False
        self.debug = False
        self.table_search = args.use_table_search

    def get_phrase_token(self, output_ids):
        new_token_len = 1
        # print("phrases", phrases_list)
        new_token = output_ids[:, -1:]
        new_token_item = new_token.item()
        if self.table_search:
            new_token_item = new_token.item()
            out_str = tokenizer.decode(output_ids[0])
            if self.debug:
                print("string:", out_str)
            if "文本" in out_str:
                self.status = Status.wenben
            elif "自车车道" in out_str:
            # if "自车车道" in out_str:
                self.status = Status.ziche
            elif "备选车道" in out_str:
                self.status = Status.beixuan
            # elif "道路类型" in out_str:
                # status = Status.daolu
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

        if new_token_len < draft_token_len + 1:
            new_token = torch.concat((new_token, fake_draft_tensor[:, new_token_len:(draft_token_len + 1)]), dim=-1)
            new_token_len = draft_token_len + 1
        # if new_token_len < 2:
            # new_token = torch.concat((new_token, fake_draft_tensor[:, new_token_len:2]), dim=-1)
            # new_token_len = 2
        elif new_token_len > (draft_token_len + 1):
            new_token = new_token[:, :(draft_token_len + 1)]
            new_token_len = (draft_token_len + 1)

        return new_token, new_token_len, None


total_time = 0
total_token = 0
avg_acc_lens = 0

count = 0
avg_acc_len = 0
total_input_len = 0
total_base_time = 0.
total_generate_steps = 0.
total_vit_time = 0.
acc_lens = dict()
for i in range(draft_token_len+1):
    acc_lens[i] = 0

total_generate_time = 0.
total_small_time = 0.
total_big_time = 0.
prefill = 0.
small_prefill = 0.
def decode_regular(self,
                   batch_size: int,
                   scfg: SamplingConfig,
                   sequence_lengths: torch.Tensor,
                   context_lengths: torch.Tensor,
                   host_context_lengths,
                   max_context_length: int,
                   beam_width: int,
                   cache_indirections: list,
                   input_ids: torch.Tensor,
                   hidden_states: torch.Tensor,
                   prompt_embedding_table: torch.Tensor,
                   tasks: torch.Tensor,
                   prompt_vocab_size: torch.Tensor,
                   ite: int,
                   sequence_limit_lengths: torch.Tensor,
                   stop_words_list,
                   bad_words_list,
                   no_repeat_ngram_size,
                   output_sequence_lengths: bool = False,
                   return_dict: bool = False,
                   encoder_output: torch.Tensor = None,
                   encoder_input_lengths: torch.Tensor = None,
                   stopping_criteria: StoppingCriteria = None,
                   logits_processor: LogitsProcessor = None,
                   cross_attention_mask: torch.Tensor = None,
                   **kwargs):
    self.stream.synchronize()
    assert(batch_size == 1)
    global total_big_time
    global total_generate_time
    global total_small_time
    prefill = 0.
    kv_cache_block_pointers = []
    host_kv_cache_block_pointers = []
    kv_cache_block_pointers_s = []
    host_kv_cache_block_pointers_s = []
    attention_mask = None
    context_logits = None
    generation_logits = []
    self.lm_head = eagle_model.base_model.lm_head
    # output_ids = [get_prefix()]
    # input_ids = torch.concat((input_ids.cuda(), get_prefix().squeeze(0)), dim=-1)
    status = 0
    self.phrase_len = 0
    self.lm_head_time = 0.
    choices = Choices()
    self.prefix_len = prefix_len
    total_accept_length = 0

    def get_outputs_dict(output_ids, total_accept_length, step):
        outputs = {}
        outputs['output_ids'] = output_ids
        outputs['total_accept_length'] = total_accept_length
        outputs['step'] = step
        if output_sequence_lengths:
            outputs['sequence_lengths'] = self.sequence_length_buffer.reshape([batch_size, beam_width])
        if self.gather_context_logits:
            outputs['context_logits'] = context_logits
        if self.gather_generation_logits:
            outputs['generation_logits'] = generation_logits
        # if self.is_medusa_mode and not self.use_context_fmha_for_generation:
            # outputs['medusa_output_tokens'] = self.medusa_output_tokens
            # outputs['accept_lengths'] = self.accept_lengths
            # if self.medusa_temperature != 0.0:
                # outputs['medusa_output_logits'] = self.medusa_output_logits
        return outputs

    benchmark_profiler = kwargs.get('benchmark_profiler', None)
    generation_phase_step_count = 0

    def profile_fn(benchmark_profiler_obj, step_count):
        if benchmark_profiler_obj is not None:
            benchmark_profiler_obj.record_cuda_event('last_token')
            benchmark_profiler_obj.record_elapsed_time(
                'first_token', 'last_token', 'generation_time')
            benchmark_profiler_obj.add_aux_info('generation_step_count',
                                                step_count)

    self.total_time = 0.
    eagle_model.ea_layer.reset_kv(0)

    next_step_tensors = None
    next_step_tensors_s = None
    total_hidden_states = None
    input_embedding = hidden_states.clone()
    total_hidden_states = None
    context_lengths_o = context_lengths.clone()
    small_sequence_lengths = sequence_lengths.clone()
    tasks_s = tasks.clone()
    encoder_input_lengths_s = encoder_input_lengths
    accept_length = 0
    if batch_size == 1:
        current_len = input_ids.shape[0]
    else:
        current_len = input_ids.shape[1]
    new_token_len = draft_token_len + 1
    # print("max_context_length", max_context_length)
    self.stream.synchronize()
    all_start = time.time()
    for step in range(0, self.max_new_tokens):
        self.stream.synchronize()
        start = time.time()
        should_stop, next_step_tensors, tasks, context_lengths, host_context_lengths, attention_mask, logits, generation_logit, encoder_input_lengths, hidden_states_output = self.handle_per_step(
            cache_indirections, step, batch_size, max_context_length,
            beam_width, input_ids, hidden_states, scfg,
            kv_cache_block_pointers, host_kv_cache_block_pointers,
            prompt_embedding_table, tasks, context_lengths,
            host_context_lengths, attention_mask, cross_attention_mask,
            prompt_vocab_size, ite, sequence_limit_lengths,
            sequence_lengths, next_step_tensors, stop_words_list,
            bad_words_list, no_repeat_ngram_size, encoder_output,
            encoder_input_lengths, stopping_criteria, logits_processor,
            **kwargs)
        if step == 0:
            if benchmark_profiler is not None:
                benchmark_profiler.record_cuda_event('first_token')

            big_new_tokens = torch.argmax(generation_logit, dim=-1).to(torch.int32)
            # NOTE: self.accept_lengths are the lengths of accepted tokens in the current step
            ### context_lengths += 1
            if prefix_len > 0 and args.use_mds:
                posterior_mask = (prefix == big_new_tokens[:, :-1]).int()
                accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
                assert(accept_length == self.prefix_len)
            big_new_tokens = big_new_tokens[:, :accept_length+1]
            accept_length = torch.Tensor([1]).to(torch.int32).cuda()

            if args.use_mds:
                self.sequence_length_buffer += draft_token_len + 1
            else:
                self.sequence_length_buffer += 1
            ## for small
            if args.use_mds:
                new_embedding = bigmodel.model.embed_tokens(big_new_tokens[:, -1:])
                input_embedding = torch.concat((input_embedding[:,1:,:], new_embedding), 1)
            total_hidden_states = hidden_states_output
            output_ids = big_new_tokens
        else:
            generation_phase_step_count = generation_phase_step_count + 1

            big_new_tokens = torch.argmax(generation_logit, dim=-1).to(torch.int32)
            # print("input_ids[:, :]", input_ids)
            # print("big_new_tokens[:, :]", big_new_tokens)

            posterior_mask = (input_ids[:, 1:] == big_new_tokens[:, :-1]).int()
            accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
            accept_length += 1
            # accept_length = torch.Tensor([1]).to(torch.int32).cuda()

            # NOTE: self.sequence_length_buffer = num_past_kv_cache (accepted) + num_medusa_tokens + 1
            big_new_tokens = big_new_tokens[:, :accept_length]
            # NOTE: self.accept_lengths are the lengths of accepted tokens in the current step
            self.accept_lengths = accept_length
            # if args.use_mds and not args.use_table_search:
                # small.accept_lengths = accept_length
            if args.use_mds or args.use_table_search:
                self.update_kv_cache_draft_token_location(batch_size, self.medusa_paths[0][:new_token_len], accept_length)
            else:
                self.sequence_length_buffer += 1

            ## for small
            if args.use_mds:
                input_embedding = torch.concat((input_embedding, bigmodel.model.embed_tokens(big_new_tokens[:, :accept_length])), 1)
                if total_hidden_states is not None:
                    total_hidden_states = torch.concat((total_hidden_states, hidden_states_output[:, :accept_length]), 1)
                else:
                    total_hidden_states = hidden_states_output[:, :accept_length]
            acc_lens[accept_length.item()-1] += 1
            total_accept_length += accept_length

            output_ids = torch.concat((output_ids, big_new_tokens), -1)
            # if accept_length < 11:
                # print("input_ids[:, :]", input_ids)
                # print("big_new_tokens[:, :]", big_new_tokens)
                # print("目前: tokens:", output_ids[0])
                # print("目前:", tokenizer.decode(output_ids[0]))
        # output_ids.item()
        self.stream.synchronize()
        end = time.time()
        if count > 1:
            if (step > 0):
                total_big_time += end - start
            else:
                prefill = end - start

        # print("目前: tokens:", output_ids[0])
        # print("目前:", tokenizer.decode(output_ids[0]))
        # 108704 is "文本"
        # should_stop = (self.end_ids in output_ids[0]) or (108704 in output_ids[0])
        should_stop = (self.end_ids in output_ids[0])
        if should_stop is not None and should_stop:
            profile_fn(benchmark_profiler, generation_phase_step_count)
            if args.use_mds:
                self.sequence_length_buffer = self.sequence_length_buffer - self.num_medusa_tokens
            final_output_ids = output_ids
            total_generate_time += end - start
            if count > 1:
                print("prefill", prefill)
                print("small_prefill", small_prefill)
                print("total_big_time", total_big_time / (count + 1))
                print("small time", total_small_time / (count + 1))
                print("total_time", (end - all_start))
            if self.mapping.is_first_pp_rank():
                if return_dict:
                    return get_outputs_dict(final_output_ids, total_accept_length, step + 1)
                else:
                    return final_output_ids
            elif self.mapping.is_last_pp_rank():
                outputs = {}
                if self.gather_context_logits:
                    outputs['context_logits'] = context_logits
                if self.gather_generation_logits:
                    outputs['generation_logits'] = generation_logits
                return outputs
            else:
                return None

        self.stream.synchronize()
        start_small = time.time()
        input_ids = big_new_tokens[:, -1:]
        # phrase_tokens, new_token_len, status = choices.get_phrase_token(output_ids)
        if new_token_len > 1 and args.use_table_search:
            input_ids = phrase_tokens
        elif args.use_mds:
            ea_logits, _, _ = eagle_model.ea_layer.topOne_genrate(
                total_hidden_states,
                input_embedding,
                head=eagle_model.base_model.lm_head,
                logits_processor=None,
                max_length=draft_token_len,
                prefill=(step == 0),
                # end_ids=self.end_ids,
            )
            # fake_draft_tensor[:,:(ea_logits.shape[0]+1)] = torch.concat((input_ids, ea_logits.unsqueeze(0)), dim=-1).to(torch.int32)
            # input_ids = fake_draft_tensor
            input_ids = torch.concat((input_ids, ea_logits.unsqueeze(0)), dim=-1).to(torch.int32)
            new_token_len = input_ids.shape[1]
            eagle_model.ea_layer.revert_kv(ea_logits.shape[0]-1)
            total_hidden_states = None
        else:
            input_ids = big_new_tokens[:, -1:]
        self.stream.synchronize()
        end = time.time()
        # print("猜 tokens:", input_ids[0])
        # print("猜:", tokenizer.decode(input_ids[0]))

        next_context = self.runtime.context_1 if step % 2 else self.runtime.context_0
        self.runtime._set_tensor(next_context, "input_ids", input_ids.squeeze(0))
        # self.runtime._set_tensor(next_context, "medusa_packed_mask", medusa_packed_mask[:,:new_token_len].detach().clone())
        # self.runtime._set_tensor(next_context, "medusa_position_offsets", medusa_position_offsets[:,:new_token_len].detach().clone())

        self.new_token_num = new_token_len
        current_len += accept_length.item()
        # print("self.new_token_num", self.new_token_num)

        if count > 1:
            if (step > 0):
                total_generate_time += end - start
                total_small_time += end - start_small
            else:
                small_prefill = end - start - prefill
        # if step == 5:
            # exit()

    assert not self.is_medusa_mode, "the custom decoder doesn't support medusa."

    profile_fn(benchmark_profiler, generation_phase_step_count)

    # final_output_ids = self.finalize_decoder(context_lengths, batch_size,
                                             # beam_width, scfg)
    final_output_ids = torch.concat(output_ids, -1)
    if self.mapping.is_first_pp_rank():
        if return_dict:
            return get_outputs_dict(final_output_ids, accept_length, step)
        else:
            return final_output_ids
    elif self.mapping.is_last_pp_rank():
        outputs = {}
        if self.gather_context_logits:
            outputs['context_logits'] = context_logits
        if self.gather_generation_logits:
            outputs['generation_logits'] = generation_logits
        return outputs
    else:
        return None


def decode(self,
           input_ids: torch.Tensor,
           input_embedding: torch.Tensor,
           context_lengths: torch.Tensor,
           sampling_config: SamplingConfig,
           prompt_embedding_table: torch.Tensor = None,
           tasks: torch.Tensor = None,
           prompt_vocab_size: torch.Tensor = None,
           stop_words_list=None,
           bad_words_list=None,
           no_repeat_ngram_size=None,
           streaming: bool = False,
           output_sequence_lengths: bool = False,
           return_dict: bool = False,
           encoder_output: torch.Tensor = None,
           encoder_input_lengths: torch.Tensor = None,
           stopping_criteria: StoppingCriteria = None,
           logits_processor: LogitsProcessor = None,
           cross_attention_mask: torch.Tensor = None,
           **kwargs):
    scfg = sampling_config
    batch_size = context_lengths.size(0)
    beam_width = scfg.num_beams
    max_context_length = torch.max(context_lengths).item()
    host_context_lengths = context_lengths.cpu()
    assert batch_size == self.batch_size, \
        "Given batch size is different from the one used in setup()," \
        "rerun the setup function with the new batch size to avoid buffer overflow."
    assert max_context_length <= self.max_context_length, \
        "Given input length is large then the one used in setup()," \
        "rerun the setup function with the new max_context_length to avoid buffer overflow."
    assert beam_width == self.beam_width, \
        "Given beam width is different from the one used in setup()," \
        "rerun the setup function with the new beam width to avoid buffer overflow."
    assert self.sink_token_length <= torch.min(context_lengths).item(), \
        "Given sink token length is larger than shortest context length," \
        "rerun the setup function with a smaller sink token length."
    ite = 0  # index of local batches, will always be 0 if pp_size = 1

    if self.remove_input_padding and input_ids.dim() == 2:
        assert input_ids.shape[
            0] == 1, "Packed 2D input must have shape [1, <sum of input lengths>]"
        input_ids = input_ids.squeeze(0)

    self.setup_decoder(input_ids, scfg, host_context_lengths)
    if not self.buffer_allocated:
        raise RuntimeError('Buffer not allocated, please call setup first!')

    sequence_limit_lengths = torch.full((batch_size, 1),
                                        self.max_seq_length,
                                        dtype=torch.int32,
                                        device=self.device)

    # Sequence_lengths for the dynamic decoder still has the input paddings.
    sequence_lengths = torch.full((batch_size * beam_width, 1),
                                  max_context_length,
                                  dtype=torch.int32,
                                  device=self.device)

    cache_indirections = [
        torch.full((
            batch_size,
            beam_width,
            self.max_attention_window_size,
        ),
                   0,
                   dtype=torch.int32,
                   device=self.device),
        torch.full((
            batch_size,
            beam_width,
            self.max_attention_window_size,
        ),
                   0,
                   dtype=torch.int32,
                   device=self.device)
    ]  # ping-pong buffers

    if self.is_medusa_mode:
        if self.quant_mode.has_kv_cache_quant():
            # Since torch does not support fp8 now, using int8 here.
            kv_cache_type = torch.int8
        else:
            kv_cache_type = self.dtype if self.paged_kv_cache else self._tensor_dtype(
                f'present_key_value_{self.first_layer}')
        self.history_max_seq_length = [max_context_length]
        self.kv_cache_updater = KVCacheUpdater()
        assert not self.cross_attention
        assert self.use_gpt_attention_plugin

        if self.paged_kv_cache:
            self.kv_cache_updater.init_paged_kv_cache(
                self.num_heads_kv, self.head_size, kv_cache_type,
                self.kv_cache_manager)
        else:
            past_key_value_list = [
                self.buffer[f'present_key_value_{i}']
                for i in range(self.first_layer, self.last_layer)
            ]
            self.kv_cache_updater.init_linear_kv_cache(
                self.num_heads_kv, self.head_size, kv_cache_type,
                past_key_value_list)

    return decode_regular(
        self,
        batch_size, scfg, sequence_lengths, context_lengths,
        host_context_lengths, max_context_length, beam_width,
        cache_indirections, input_ids, input_embedding,
        prompt_embedding_table, tasks, prompt_vocab_size, ite,
        sequence_limit_lengths, stop_words_list, bad_words_list,
        no_repeat_ngram_size, output_sequence_lengths, return_dict,
        encoder_output, encoder_input_lengths, stopping_criteria,
        logits_processor, cross_attention_mask, **kwargs)


def generate(self,
             batch_input_ids: List[torch.Tensor],
             input_embedding: torch.Tensor = None,
             sampling_config: Optional[SamplingConfig] = None,
             ptuning_args = None,
             prompt_table_path = None,
             prompt_tasks: Optional[str] = None,
             lora_uids: Optional[list] = None,
             streaming: bool = False,
             stopping_criteria: Optional[StoppingCriteria] = None,
             logits_processor: Optional[LogitsProcessor] = None,
             medusa_choices: Optional[List[List[int]]] = None,
             **kwargs) -> Union[torch.Tensor, dict]:
    # Use sampling_config like HF's generation_config
    if sampling_config is None:
        sampling_config = SamplingConfig(end_id=None, pad_id=None)
    else:
        sampling_config = copy.deepcopy(sampling_config)
    sampling_config.update(**kwargs)
    self._check_inputs(batch_input_ids, sampling_config)

    batch_size = len(batch_input_ids)
    batch_input_ids, input_lengths = self._prepare_inputs(
        batch_input_ids, sampling_config.pad_id)

    self.session.setup(
        batch_size=batch_size,
        max_context_length=input_lengths.max().item(),
        max_new_tokens=sampling_config.max_new_tokens,
        beam_width=sampling_config.num_beams,
        max_attention_window_size=sampling_config.max_attention_window_size,
        sink_token_length=sampling_config.sink_token_length,
        lora_manager=self.lora_manager,
        lora_uids=lora_uids,
        medusa_choices=medusa_choices)

    batch_input_ids = batch_input_ids.cuda()
    input_lengths = input_lengths.cuda()
    ptuning_kwargs = ptuning_args
    # ptuning_kwargs = self._prepare_ptuning(prompt_table_path, prompt_tasks,
                                           # batch_size)
    external_stream = torch.cuda.current_stream()
    torch.cuda.set_stream(self.session.stream)
    outputs = decode(
        self.session,
        batch_input_ids,
        input_embedding,
        input_lengths,
        sampling_config,
        stop_words_list=sampling_config.stop_words_list,
        bad_words_list=sampling_config.bad_words_list,
        output_sequence_lengths=sampling_config.output_sequence_lengths,
        return_dict=sampling_config.return_dict,
        streaming=streaming,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
        **ptuning_kwargs)
    torch.cuda.set_stream(external_stream)
    if sampling_config.return_dict:
        if streaming:
            outputs = (self._prepare_outputs(curr_outputs, input_lengths)
                       for curr_outputs in outputs)
        else:
            outputs = self._prepare_outputs(outputs, input_lengths)
    return outputs

with torch.inference_mode():
    # while count < args.benchmark_steps:
    for idx, data in enumerate(dataset):
        count = idx
        (input_ids, image_tensor, image_sizes, prompt) = data
        print("prompt:", prompt)
        print("input_ids:", input_ids.shape)

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if prefix_len > 0:
            input_ids = torch.concat((input_ids.cuda(), prefix), dim=-1)

        big_runner.session.stream.synchronize()
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
        big_runner.session.stream.synchronize()
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
        input_ids, ptuning_args = setup_fake_prompts(input_embedding)
        # print("ptuning_args", ptuning_args)

        big_runner.session.stream.synchronize()
        start = time.time()
        # outputs = {
            # 'output_ids': [input_ids],
        # }

        # outputs = big_runner.generate(
        outputs = generate(
            big_runner,
            batch_input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id,
            # temperature=temperature,
            # top_k=top_k,
            # top_p=top_p,
            # num_beams=num_beams,
            input_embedding=input_embedding,
            # prompt_table_path=prompt_table_path,
            ptuning_args=ptuning_args,
            # stop_words_list=stop_words_list,
            # bad_words_list=bad_words_list,
            # early_stopping=early_stopping,
            # stopping_criteria=stopping_criteria,
            # streaming=streaming,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=medusa_choices,
        )
        big_runner.session.stream.synchronize()
        end = time.time()

        # print(outputs)
        if prefix_len > 0:
            new_token = outputs['sequence_lengths'] - input_token_len + prefix_len
        else:
            new_token = outputs['sequence_lengths'] - input_token_len

        generate_steps = outputs['step']
        # avg_acc_len = outputs['total_accept_length'] * 1. / generate_steps
        avg_acc_len = outputs['total_accept_length'] * 1. / (generate_steps - 1)
        # generation_logits = outputs['generation_logits']
        outputs = outputs['output_ids']
        print("outputs_id:", outputs)
        seen_tokens_set = set(outputs.cpu().numpy().flatten()) | seen_tokens_set

        if False:
            outputs = outputs[:, input_token_len:]
        # print("generation_logits", generation_logits.shape)
        print("Step", idx, ":", end - start, "s")
        print("Outputs:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        print("generate ite num:", generate_steps)
        print("outputs.numel():", outputs.numel())
        print("Total new token:", new_token)
        print("Avg acc len:", avg_acc_len)

        if (count > 1):
            total_time += (end - start)
            total_token += new_token
            avg_acc_lens += avg_acc_len
            total_input_len += input_token_len
            total_base_time += (base_end - base_start)
            total_vit_time += (vit_end - vit_start)
            total_generate_steps += generate_steps
            print("avg_acc_lens:", avg_acc_lens / (count - 1))
        if (idx == args.benchmark_steps):
            break
count -= 1
print("Total steps:", args.benchmark_steps)
print("Avg generate steps:", total_generate_steps / count)
print("Batch size:", args.batch_size)
print("Input len:", total_input_len / count)
print("Avg output len:", total_token / count)
print("Avg acc len:", avg_acc_lens / count - 1)
print("Avg tokens / step:", avg_acc_lens / count)
print("Avg vit time:", total_vit_time / count)
print("Avg total time:", (total_time) / count)
print("benchmark_steps:", count)
print("BASE TPS:", total_token / total_base_time)
print("TPS:", total_token / total_time)
print("acc_lens:", acc_lens)
print("avg generate time", total_generate_time / (count))
print("seen_tokens_set", seen_tokens_set)

