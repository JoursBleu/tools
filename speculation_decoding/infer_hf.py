from datasets import load_dataset
from transformers import AutoTokenizer
from lpex_llm.inference import InferenceModelTrtllm
from lpex_llm.inference import LpexTrtllmConfig




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--max_input_len', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--run_hf', action="store_true", default=False)
    parser.add_argument('--use_sp', action="store_true", default=False)
    parser.add_argument('--benchmark', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_dataset_json', type=str, default="/lpai/volumes/cloudmodel-muses/zhangkuo/data/vlm_sft_v0.4.5_yq_shuf_wo_image.json")

    args, unknown = parser.parse_known_args()
    return args




if __name__ == '__main__':
    args = parse_arguments()
    ds = load_dataset('json', data_files=args.benchmark_dataset_json)

    if args.tokenizer is None:
        args.tokenizer = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    lpex_config = LpexTrtllmConfig()
    lpex_config.cache_dir = "/lpai/volumes/cloudmodel-muses/lt/lpex_cache"
    lpex_config.dtype = "float16"
    lpex_config.max_batch_size = args.batch_size
    lpex_config.max_input_len = 2048
    lpex_config.max_output_len = 2048
    lpex_config.max_seq_length = lpex_config.max_input_len + lpex_config.max_output_len
    lpex_config.max_num_tokens = lpex_config.max_seq_length * lpex_config.max_batch_size
    lpex_config.use_cached_model = False
    lpex_config.use_weight_only = False

    model = InferenceModelTrtllm.from_hf_model_dir(
        args.model_dir,
        lpex_config=lpex_config,
        tokenizer=tokenizer,
    )

    model



