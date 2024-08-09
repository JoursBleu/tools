import json

from transformers import AutoTokenizer
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

qwen2_config = Qwen2Config.from_pretrained("/lpai/volumes/lpai-demo-muses/lt/models/Qwen1.5-0.5B-Chat/config.json")
qwen2_model = Qwen2Model(qwen2_config).eval()

tokenizer = AutoTokenizer.from_pretrained("/lpai/volumes/lpai-demo-muses/lt/models/Qwen1.5-0.5B-Chat")
# prepare post image prompt
fp = open("/lpai/volumes/lpai-demo-muses/lt/sparsedrive/data/nuScenes_train_llava.json", 'r')
prompts = json.load(fp)
prompts_dict = dict()
max_len = 0
index = 0
for ele in prompts:
    try:
        index += 1
        # 默认只有一轮对话
        user = ele['conversations'][0]['value'].split("<image>\n")[1]
        assistant = ele['conversations'][1]['value']
        pos_image = user + "<|im_end|>"
        # pos_image_id = tokenizer.encode(pos_image, return_tensors="pt", padding='max_length', max_length=1600)
        # # pos_image = user + "<|im_end|>\n<|im_start|>ASSISTANT:" + assistant + "<|im_end|>"
        # # pos_image_id = tokenizer.encode(pos_image, return_tensors="pt", padding='max_length', max_length=2400)
        # pos_image_emb = qwen2_model.embed_tokens(pos_image_id)
        pos_image_id = tokenizer.encode(pos_image, padding='max_length', max_length=1600)
        prompts_dict[ele['id']] = pos_image_id
        # if (max_len < pos_image_id.shape[1]):
            # max_len = pos_image_id.shape[1]
        print("index", index)
        # print("rank:", torch.distributed.get_rank(), " index:", index)
        # torch.distributed.barrier()
    except Exception as e:
        breakpoint()
prompts_json = json.dumps(prompts_dict)
file = open("/lpai/volumes/lpai-demo-muses/lt/sparsedrive/data/prompts_embedding.json", "w")
file.write(prompts_json)
print("max_len", max_len)
exit()