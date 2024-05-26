import torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--big_model_dir', type=str, default="/lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02")
    parser.add_argument('--small_model_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19")
    parser.add_argument('--output_dir', type=str, default="/mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-small")
    # "/lpai/volumes/cloudmodel-muses/lt/data/llava_data"

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

ea_layer_state_dict = torch.load(args.small_model_dir+"/pytorch_model.bin", map_location=torch.device("cuda"))

config=AutoConfig.from_pretrained(args.small_model_dir)

model=AutoModelForCausalLM.from_config(config)

print("model", model)

new_model = Qwen2ForCausalLM.from_pretrained("/lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02")

model.fc.weight.data=ea_layer_state_dict["fc.weight"]
model.model.embed_tokens.weight.data=ea_layer_state_dict["embed_tokens.weight"]
model.model.layers[0].self_attn.q_proj.weight.data=ea_layer_state_dict["layers.0.self_attn.q_proj.weight"]
model.model.layers[0].self_attn.k_proj.weight.data=ea_layer_state_dict["layers.0.self_attn.k_proj.weight"]
model.model.layers[0].self_attn.v_proj.weight.data=ea_layer_state_dict["layers.0.self_attn.v_proj.weight"]
model.model.layers[0].self_attn.o_proj.weight.data=ea_layer_state_dict["layers.0.self_attn.o_proj.weight"]
model.model.layers[0].mlp.gate_proj.weight.data=ea_layer_state_dict["layers.0.mlp.gate_proj.weight"]
model.model.layers[0].mlp.up_proj.weight.data=ea_layer_state_dict["layers.0.mlp.up_proj.weight"]
model.model.layers[0].mlp.down_proj.weight.data=ea_layer_state_dict["layers.0.mlp.down_proj.weight"]
model.model.layers[0].post_attention_layernorm.weight.data=ea_layer_state_dict["layers.0.post_attention_layernorm.weight"]
model.lm_head.weight.data = new_model.lm_head.weight.data


model.save_pretrained(args.output_dir)
