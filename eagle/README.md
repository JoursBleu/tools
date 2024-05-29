export PYTHONPATH=/lpai/volumes/cloudmodel-muses/lt/EAGLE:/lpai/volumes/cloudmodel-muses/lt/transformers/src:/lpai/volumes/cloudmodel-muses/lt/TensorRT-LLM-v080:/lpai/volumes/cloudmodel-muses/lt/llm_factory


pip3 install tensorrt-9.2.0.post12.dev5-cp310-none-linux_x86_64.whl

## covnert target

cd /lpai/volumes/cloudmodel-muses/lt/Qwen-TensorRT-LLM/examples/qwen2

python3 build.py --hf_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --dtype float16 --remove_input_padding --gpt_attention_plugin float16 --gemm_plugin float16 --use_weight_only --weight_only_precision int4 --output_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-org --max_num_tokens 8192 --enable_context_fmha --max_batch_size 16 |tee eee.log


## convert small

/mnt/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b-small/ffn512head4/

# convert eagle small ckpt to hf llama ckpt

python3 /lpai/volumes/cloudmodel-muses/lt/tools/eagle/convert_weight.py --big_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --output_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-small


# convert hf llama ckpt to trtllm ckpt

/lpai/volumes/cloudmodel-muses/lt/TensorRT-LLM/v0.8.0_0522.patch

cd /lpai/volumes/cloudmodel-muses/lt/TensorRT-LLM/examples/llama

python3 convert_checkpoint.py --model_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4-llama --output_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4-llama-trt

# build trtllm engine

trtllm-build --checkpoint_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4-llama-trt --output_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4-llama-engine/ --gemm_plugin float16 --paged_kv_cache disable --max_input_len 2048 --max_output_len 512 --max_num_tokens 2048 --max_batch_size 16

# build trt engine

/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/prefill.onnx --saveEngine=trt-eagle/prefill.trt --minShapes=hidden_states:1x1x2048,inputs_embeds:1x1x2048,position_ids:1x1 --optShapes=hidden_states:1x600x2048,inputs_embeds:1x600x2048,position_ids:1x600 --maxShapes=hidden_states:1x800x2048,inputs_embeds:1x800x2048,position_ids:1x800

/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/context.onnx --saveEngine=trt-eagle/context.trt --minShapes=hidden_states:1x1x2048,inputs_embeds:1x1x2048,past_key_values:2x1x4x600x128,position_ids:1x1 --optShapes=hidden_states:1x16x2048,inputs_embeds:1x16x2048,past_key_values:2x1x4x680x128,position_ids:1x16 --maxShapes=hidden_states:1x48x2048,inputs_embeds:1x48x2048,past_key_values:2x1x4x800x128,position_ids:1x48


/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/generate.onnx --saveEngine=trt-eagle/generate.trt --minShapes=past_key_values:2x1x4x1x128 --optShapes=past_key_values:2x1x4x600x128 --maxShapes=past_key_values:2x1x4x800x128


/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/prefill.onnx --saveEngine=trt-eagle/prefill.trt --minShapes=hidden_states:1x1x2048,inputs_embeds:1x1x2048,position_ids:1x1 --optShapes=hidden_states:1x600x2048,inputs_embeds:1x600x2048,position_ids:1x600 --maxShapes=hidden_states:1x800x2048,inputs_embeds:1x800x2048,position_ids:1x800

/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/context.onnx --saveEngine=trt-eagle/context.trt --minShapes=hidden_states:1x1x2048,inputs_embeds:1x1x2048,past_key_values:2x1x16x600x128,position_ids:1x1 --optShapes=hidden_states:1x16x2048,inputs_embeds:1x16x2048,past_key_values:2x1x16x680x128,position_ids:1x16 --maxShapes=hidden_states:1x48x2048,inputs_embeds:1x48x2048,past_key_values:2x1x16x800x128,position_ids:1x48


/usr/local/tensorrt/bin/trtexec --onnx=trt-eagle/generate.onnx --saveEngine=trt-eagle/generate.trt --minShapes=past_key_values:2x1x16x1x128 --optShapes=past_key_values:2x1x16x600x128 --maxShapes=past_key_values:2x1x16x800x128


# base without eagle

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19

		Total steps: 100
		Batch size: 1
		Input len: 244.43434343434345
		Avg output len: 56.95959595959596
		Avg acc len: 0.0
		Avg token/step: 1.0
		Avg vit time: 0.009110910723907779
		benchmark_steps: 99
		BASE TPS: 215015275.05454546
		TPS: 33.956882150580725

# base eagle

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19

		Total steps: 100
		Batch size: 1
		Input len: 244.43434343434345
		Avg output len: 57.707070707070706
		Avg acc len: tensor(7.2888, device='cuda:0')
		Avg token/step: tensor(8.2888, device='cuda:0')
		Avg vit time: 0.009311184738621567
		benchmark_steps: 99
		BASE TPS: 230404411.07692307
		TPS: 144.0485884400107

# trt base without eagle

cmd:




# trt context4gen without sp

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_context4gen.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-context/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp|tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[33.8990]], device='cuda:0')
		Avg acc len: tensor([1.], device='cuda:0')
		Avg tokens / step: tensor([2.], device='cuda:0')
		Avg vit time: 0.009584968740289862
		Avg total time: 0.34890622081178607
		benchmark_steps: 99
		BASE TPS: tensor([[1.3406e+08]], device='cuda:0')
		TPS: tensor([[99.9024]], device='cuda:0')


# trt context4gen with table-search

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_context4gen.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-context/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_table_search |tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[41.2323]], device='cuda:0')
		Avg acc len: tensor([4.0694], device='cuda:0')
		Avg tokens / step: tensor([5.0694], device='cuda:0')
		Avg vit time: 0.00960053097118031
		Avg total time: 0.15135504501034516
		benchmark_steps: 99
		BASE TPS: tensor([[1.5151e+08]], device='cuda:0')
		TPS: tensor([[290.8713]], device='cuda:0')


# trt context4gen with sp & table-search

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_context_org_small.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-context/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_table_search |tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[56.5758]], device='cuda:0')
		Avg acc len: tensor([5.3536], device='cuda:0')
		Avg tokens / step: tensor([6.3536], device='cuda:0')
		Avg vit time: 0.010370225617379852
		Avg total time: 0.1946053095538207
		benchmark_steps: 99
		BASE TPS: tensor([[1.4501e+08]], device='cuda:0')
		TPS: tensor([[307.0846]], device='cuda:0')


# trt base

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-base/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp |tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[56.5152]], device='cuda:0')
		Avg acc len: tensor([0.], device='cuda:0')
		Avg tokens / step: tensor([1.], device='cuda:0')
		Avg vit time: 0.011312761692085652
		Avg total time: 0.2915088865492079
		benchmark_steps: 99
		BASE TPS: tensor([[2.7287e+08]], device='cuda:0')
		TPS: tensor([[201.6986]], device='cuda:0')


# trt medusa with table-search

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_mds --use_table_search |tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[57.5152]], device='cuda:0')
		Avg acc len: tensor([2.6493], device='cuda:0')
		Avg tokens / step: tensor([3.6493], device='cuda:0')
		Avg vit time: 0.0097628217754942
		Avg total time: 0.25528860092163086
		benchmark_steps: 99
		BASE TPS: tensor([[2.5139e+08]], device='cuda:0')
		TPS: tensor([[234.2530]], device='cuda:0')



# trt sp with eagle

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_mds |tee eee.log


		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[57.7273]], device='cuda:0')
		Avg acc len: tensor([5.9471], device='cuda:0')
		Avg tokens / step: tensor([6.9471], device='cuda:0')
		Avg vit time: 0.00940054835695209
		Avg total time: 0.1861219382045245
		benchmark_steps: 99
		BASE TPS: tensor([[2.5775e+08]], device='cuda:0')
		TPS: tensor([[326.6570]], device='cuda:0')


# trtllm sp with trt engle engine

cmd:

CUDA_VISIBLE_DEVICES=6 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4/ --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0 --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0/ --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b-mds-engine/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.6.0_bus_0/navi_llava_train.json --use_sp --mds_len 20 --small_engine_dir trt-mds/ffn512head4.trt |tee eee.log


		Total steps: 1000
		Avg generate steps: 3.197394789579158
		Batch size: 1
		Input len: 607.0
		Avg output len: tensor([[41.6573]], device='cuda:0')
		Avg acc len: tensor([11.0123], device='cuda:0')
		Avg tokens / step: tensor([12.0123], device='cuda:0')
		Avg vit time: 0.012190226801411661
		Avg total time: 0.05420246320162603
		benchmark_steps: 998
		BASE TPS: tensor([[1.6101e+08]], device='cuda:0')
		TPS: tensor([[768.5502]], device='cuda:0')
		acc_lens: {0: 14, 1: 15, 2: 10, 3: 56, 4: 78, 5: 22, 6: 8, 7: 4, 8: 23, 9: 26, 10: 25, 11: 25, 12: 19, 13: 51, 14: 125, 15: 160, 16: 373, 17: 106, 18: 37, 19: 58, 20: 962}
		avg generate time 0.020013214829928412


# trtllm sp with trt engle engine with small lm_head

cmd:

CUDA_VISIBLE_DEVICES=6 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4/ --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0 --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0/ --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b-mds-engine/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.6.0_bus_0/navi_llava_train.json --use_sp --mds_len 20 --small_engine_dir trt-mds/ffn512head4.trt --use_small_head |tee eee.log


		Total steps: 1000
		Avg generate steps: 3.1923847695390783
		Batch size: 1
		Input len: 607.0
		Avg output len: tensor([[41.6663]], device='cuda:0')
		Avg acc len: tensor([16.8272], device='cuda:0')
		Avg tokens / step: tensor([17.8272], device='cuda:0')
		Avg vit time: 0.014057678306747774
		Avg total time: 0.04875197152575415
		benchmark_steps: 998
		BASE TPS: tensor([[1.2584e+08]], device='cuda:0')
		TPS: tensor([[854.6594]], device='cuda:0')
		acc_lens: {0: 20, 1: 15, 2: 9, 3: 47, 4: 78, 5: 21, 6: 7, 7: 4, 8: 23, 9: 23, 10: 23, 11: 24, 12: 18, 13: 50, 14: 130, 15: 160, 16: 375, 17: 108, 18: 37, 19: 57, 20: 963}
		avg generate time 0.021830198521126727
		seen_tokens_set {117507, 111363, 85254, 11, 1294, 15, 16, 39953, 106772, 22, 24, 25, 26, 30767, 101178, 58, 111687, 100167, 109131, 35926, 7259, 151643, 53599, 57191, 86119, 100714, 17259, 100205, 104307, 118, 102540, 35727, 5265, 112018, 103583, 31905, 99257, 100029, 100806, 8903, 104399, 42192, 56278, 220, 20450, 108005, 107500, 44793, 113658, 105212}



# yiqiang

cmd:

	CUDA_VISIBLE_DEVICES=1 python3 eval_eagle_llava_trt_mds_yiqiang.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4/ --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0 --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b_siglip384_960_cdp320_task_sft_v0.6.0_bus_0 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/EAGLE-base/small_v3/ffn512head4-llama-engine/ --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/llava_qwen1.8b-mds-engine --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_mds | tee llm.log


	CUDA_VISIBLE_DEVICES=1 python3 eval_eagle_llava_trt_mds_yiqiang.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0525-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_mds | tee llm.log