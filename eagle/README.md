export PYTHONPATH=/lpai/volumes/cloudmodel-muses/lt/EAGLE:/lpai/volumes/cloudmodel-muses/lt/transformers/src:/lpai/volumes/cloudmodel-muses/lt/TensorRT-LLM-v080:/lpai/volumes/cloudmodel-muses/lt/llm_factory







# base without eagle

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --max_input_len 64 --max_new_tokens 4610 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19

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

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --max_input_len 64 --max_new_tokens 4610 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19

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


# trt context4gen without sp

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds_context.py --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp|tee eee.log


# trt medusa with table-search

cmd:

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp --use_table_search |tee eee.log

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

CUDA_VISIBLE_DEVICES=0 python3 eval_eagle_llava_trt_mds.py --eagle_small_model_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/model_19 --base_model_dir /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02/ --tokenizer /lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02 --small_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-small/0503-engine --big_engine_dir /mnt/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02-trt-mds/ --max_input_len 64 --max_new_tokens 512 --benchmark --batch_size 1 --benchmark_steps 100 --benchmark_dataset_json /mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json --use_sp |tee eee.log

		Total steps: 100
		Batch size: 1
		Input len: 819.4343434343434
		Avg output len: tensor([[67.9192]], device='cuda:0')
		Avg acc len: tensor([5.9471], device='cuda:0')
		Avg tokens / step: tensor([6.9471], device='cuda:0')
		Avg vit time: 0.009291824668344825
		Avg total time: 0.2630866392694338
		benchmark_steps: 99
		BASE TPS: tensor([[1.8554e+08]], device='cuda:0')
		TPS: tensor([[267.6146]], device='cuda:0')

