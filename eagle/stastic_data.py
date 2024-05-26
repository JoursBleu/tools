import argparse
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default="/lpai/volumes/cloudmodel-muses/lt/models/wdf-llava-pretrain-ckpts-24-05-03-02")
    parser.add_argument('--benchmark_dataset_json', type=str, default="/mnt/volumes/cloudmodel-muses/lt/data/Damo/dm_v0.1.3_0/navi_llava_test.json")
    # "/lpai/volumes/cloudmodel-muses/lt/data/llava_data"

    args, unknown = parser.parse_known_args()
    return args

tags = {
    'empty': [],
}

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.benchmark_dataset_json, 'r') as f:
        questions = json.load(f)

    for idx, ele in enumerate(questions):
        if len(ele['tag']['negative']) > 0:
            print(idx, ele['tag'])
        if len(ele['tag']['normal']) > 1:
            print(idx, ele['tag'])
        elif len(ele['tag']['normal']) == 0:
            tags['empty'].append(ele['conversations'][1]['value'])
        else:
            # print(idx, ele['tag']['normal'])
            if ele['tag']['normal'][0] in tags:
                tags[ele['tag']['normal'][0]].append(ele['conversations'][1]['value'])
            else:
                tags[ele['tag']['normal'][0]] = []
                tags[ele['tag']['normal'][0]].append(ele['conversations'][1]['value'])

    for key in tags:
        # print("key:", key)
        # print("len:", len(tags[key]))
        for ele in tags[key]:
            print(key, ":\t", ele)
