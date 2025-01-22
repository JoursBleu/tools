import gc
import time
import torch
import transformers

from datasets import load_from_disk
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

template_0shot = open('/lpai/volumes/lpai-demo-muses/lt/LongBench/prompts/0shot.txt', encoding='utf-8').read()

tokenizer = AutoTokenizer.from_pretrained("/lpai/volumes/lpai-demo-muses/lt/models/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
            "/lpai/volumes/lpai-demo-muses/lt/models/Qwen2.5-7B-Instruct",
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
            # load_in_8bit=True,
            device_map="auto",
        )
model = model.eval() # .cuda().half()

# data = load_from_disk(
    # f"data/longbench/{dataset[0]}"
# )
data = load_dataset('THUDM/LongBench-v2', split='train')

system = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the question below.\n\n'
text = '<text>\n$DOC$\n</text>\n\n'
question = 'What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nFormat your response as follows: "The correct answer is (insert answer here)".\n<|im_end|>\n'

system_id = tokenizer(system, return_tensors='pt').to("cuda")
_, system_len = system_id['input_ids'].shape

max_new_tokens = 128
page_size = 4096
max_len = 120000

# for ele in data:
    # context = ele['context']
    # prompt = text.replace('$DOC$', context.strip())
    # input_ids = tokenizer.encode(prompt)
    # print("length", ele["length"], "token num", len(input_ids))
# breakpoint()

for ele in data:
    # if ele["length"] == "long":
        # continue
    context = ele['context']
    prompt = text.replace('$DOC$', context.strip())

    # input_ids = tokenizer.encode(prompt)
    # if len(input_ids) > max_len:
        # input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        # prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    choice_question = question.replace('$Q$', ele['question'].strip()).replace('$C_A$', ele['choice_A'].strip()).replace('$C_B$', ele['choice_B'].strip()).replace('$C_C$', ele['choice_C'].strip()).replace('$C_D$', ele['choice_D'].strip())
    input_ids = tokenizer(system+prompt+choice_question, return_tensors='pt').to("cuda")
    question_id = tokenizer(choice_question, return_tensors="pt").to("cuda")
    _, question_len = question_id['input_ids'].shape

    # _, input_len_org = input_ids["input_ids"].shape
    # # print("input_len_org", input_len_org)
    # torch.cuda.synchronize()
    # start = time.time()
    # output_org = model.generate(**input_ids, max_new_tokens=max_new_tokens, cache_implementation="offloaded")
    # # output_org = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True)
    # response = tokenizer.decode(output_org[0, input_len_org:])
    # torch.cuda.synchronize()
    # end = time.time()
    # print("time", end - start)
    # print("response org", response)
    # print("answer", ele['answer'])
    # continue
    # breakpoint()

    # first = True
    # total_context = system_id.copy()
    # torch.cuda.synchronize()
    # start = time.time()
    # for passage in ele['context'].split('Passage'):
        # if len(passage) > 0:
            # input_str = 'Passage' + passage
            # input_ids = tokenizer(input_str, return_tensors='pt', padding=True, padding_side='left').to("cuda")
            # total_context['input_ids'] = torch.cat([total_context['input_ids'], input_ids['input_ids']], dim=1)
            # total_context['attention_mask'] = torch.cat([total_context['attention_mask'], input_ids['attention_mask']], dim=1)
            # input_ids['input_ids'] = torch.cat([system_id['input_ids'], input_ids['input_ids']], dim=1)
            # input_ids['attention_mask'] = torch.cat([system_id['attention_mask'], input_ids['attention_mask']], dim=1)

            # output = model.generate(**input_ids, max_new_tokens=1, return_dict_in_generate=True)
            # # _, input_len = input_ids['input_ids'].shape
            # # response = tokenizer.decode(output["sequences"][0, input_len:])
            # # print("response eee", response)
            # if first:
                # past_key_values = output["past_key_values"]
                # first = False
            # else:
                # _, seq_length = input_ids['input_ids'].shape
                # output["past_key_values"].slice(system_len, seq_length)
                # past_key_values.concat(output["past_key_values"])

    # total_context['input_ids'] = torch.cat([total_context['input_ids'], question_id['input_ids']], dim=1)
    # total_context['attention_mask'] = torch.cat([total_context['attention_mask'], question_id['attention_mask']], dim=1)
    # output = model.generate(**total_context, max_new_tokens=max_new_tokens, past_key_values=past_key_values, do_sample=True)
    # _, input_len = total_context['input_ids'].shape
    # response = tokenizer.decode(output[0, input_len:])
    # torch.cuda.synchronize()
    # end = time.time()
    # print("time", end - start)
    # print("response", response)

    first = True
    total_context = system_id.copy()
    torch.cuda.synchronize()
    start = time.time()
    input_ids_all = tokenizer(prompt, return_tensors='pt').to("cuda")
    _, seqlen = input_ids_all['input_ids'].shape
    total_context['input_ids'] = torch.cat([total_context['input_ids'], input_ids_all['input_ids']], dim=1)
    total_context['attention_mask'] = torch.cat([total_context['attention_mask'], input_ids_all['attention_mask']], dim=1)
    total_context['input_ids'] = torch.cat([total_context['input_ids'], question_id['input_ids']], dim=1)
    total_context['attention_mask'] = torch.cat([total_context['attention_mask'], question_id['attention_mask']], dim=1)

    current = 0
    question_id = 0
    while current < seqlen:
        if question_id < 143:
            continue
        current_end = min(current + page_size, seqlen)
        input_ids['input_ids'] = input_ids_all['input_ids'][:, current:current_end]
        input_ids['attention_mask'] = input_ids_all['attention_mask'][:, current:current_end]
        input_ids['input_ids'] = torch.cat([system_id['input_ids'], input_ids['input_ids']], dim=1)
        input_ids['attention_mask'] = torch.cat([system_id['attention_mask'], input_ids['attention_mask']], dim=1)
        current = current_end

        output = model.generate(**input_ids, max_new_tokens=1, return_dict_in_generate=True)
        # _, input_len = input_ids['input_ids'].shape
        # response = tokenizer.decode(output["sequences"][0, input_len:])
        # print("response eee", response)
        if first:
            past_key_values = output["past_key_values"]
            first = False
        else:
            _, seq_length = input_ids['input_ids'].shape
            output["past_key_values"].slice(system_len, seq_length)
            past_key_values.concat(output["past_key_values"])

    print("seqlen:", seqlen, "\tpage_size:", page_size, "\tpage num:", seqlen // page_size + 1)

    output = model.generate(**total_context, max_new_tokens=max_new_tokens, past_key_values=past_key_values, do_sample=True)
    _, input_len = total_context['input_ids'].shape
    response = tokenizer.decode(output[0, input_len:])
    torch.cuda.synchronize()
    end = time.time()
    print("question_id", question_id, "time", end - start, "\tanswer", ele['answer'])
    print("response", response)
    print("-------------------------------------------------------")

    # generation_config, model_kwargs = model._prepare_generation_config(generation_config=None)
    # input_ids, model_input_name, model_kwargs = model._prepare_model_inputs(
        # input_ids, generation_config.bos_token_id, model_kwargs
    # )
    # batch_size = input_ids.shape[0]
    # kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
    # model._prepare_special_tokens(generation_config, kwargs_has_attention_mask)
    # model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
        # input_ids, generation_config, model_kwargs
    # )
    # model_kwargs["num_logits_to_keep"] = 1
    # user_defined_cache = model_kwargs.get("past_key_values")
    # model_kwargs["use_cache"] = generation_config.use_cache
    # model_kwargs = model._get_initial_cache_position(input_ids, model_kwargs)

    # breakpoint()
    # print(tokenizer.decode(input_ids[0]))
    # input_len = len(input_ids[0])
    # while True:
        # breakpoint()
        # print(tokenizer.decode(input_ids[0, input_len:]))
        # model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # outputs = model(**model_inputs, return_dict=True, output_attentions=False)
        # next_token_logits = outputs.logits[:, -1, :].clone().float()
        # next_tokens = torch.argmax(next_token_logits, dim=-1)
        # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        # model_kwargs = model._update_model_kwargs_for_generation(
            # outputs,
            # model_kwargs,
            # is_encoder_decoder=model.config.is_encoder_decoder,
        # )
        # outputs = None
        # model_inputs = None
        # gc.collect()


breakpoint()


