import argparse
from datasets import load_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str, default=None)
    parser.add_argument('--dump_len', type=int, default=100)

    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

dataset = load_dataset('json', data_files=args.dataset_json)['train']
for i in range(args.dump_len):
    # dataset[i]['conversations'][1]['value']
    print(dataset[i])