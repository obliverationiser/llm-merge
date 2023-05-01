import os
import gc
import sys
import json
import shutil
import resource

import safetensors.torch as sft

from itertools import zip_longest

model_1_weight = 1
model_2_weight = 1

model_1_folder = 'models/OPT-350M-Erebus'
model_2_folder = 'models/OPT-350M-Nerys-v2-100MB'

merged_model_folder = f'models/OPT-350M-{model_1_weight}Erebus-{model_2_weight}Nerys-qkv'

torch_map_location = 'cpu'

save_type = 'pytorch'

shard_size = '2G'

if save_type == 'pytorch':
    save_model_prefix = 'pytorch_model'
    save_model_ext = 'bin'

if save_type == 'safetensors':
    save_model_prefix = 'model'
    save_model_ext = 'safetensors'

if (os.path.exists(merged_model_folder)):
    if len(os.listdir(merged_model_folder)) != 0:
        shutil.rmtree(merged_model_folder)
        # raise Exception(f'Non empty directory "{merged_model_folder}" already exists')

model_1_ratio = model_1_weight / (model_1_weight + model_2_weight)
model_2_ratio = model_2_weight / (model_1_weight + model_2_weight)

print(f"[*] Merging models\n\t({round(model_1_ratio * 100, 2)} %) {model_1_folder.split('/')[-1]}\n\t({round(model_2_ratio * 100, 2)} %) {model_2_folder.split('/')[-1]}\n")


def parse_size(size_string):
    if type(size_string) == str:
        match size_string[-1]:
            case 'T':
                return int(size_string[:-1]) * 1e+12
            case 'G':
                return int(size_string[:-1]) * 1e+9
            case 'M':
                return int(size_string[:-1]) * 1e+6
            case 'K':
                return int(size_string[:-1]) * 1e+3
            case 'B':
                return int(size_string[:-1])

    return size_string


def format_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def save_model(shard_size=-1):
    if shard_size is None or shard_size <= 0:
        print(f'[*] Saving model: {merged_model_folder}/{save_model_prefix}.{save_model_ext}')

        if save_type == 'safetensors':
            sft.save_file(merged_model, f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}', metadata={'format': 'pt'})
        else:
            sft.torch.save(merged_model, f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}')
    elif parse_size(shard_size) >= shard_size:
        shard_size = 0

        print(f'[*] Saving shard: {merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}')

        if save_type == 'safetensors':
            sft.save_file(merged_model, f'{merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}', metadata={'format': 'pt'})
        else:
            sft.torch.save(merged_model, f'{merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}')


model_1_files = [file for file in os.listdir(model_1_folder) if file.endswith('.bin') or file.endswith('.safetensors')]
model_2_files = [file for file in os.listdir(model_2_folder) if file.endswith('.bin') or file.endswith('.safetensors')]

model_1_files.sort()
model_2_files.sort()

max_files_length = len(model_1_files) if len(model_1_files) > len(model_2_files) else len(model_2_files)

model_files = [file for file in zip_longest(model_1_files, model_2_files)]

model_size_bytes = 0

for file in model_1_files:
    model_size_bytes += os.path.getsize(f'{model_1_folder}/{file}')

bin_weight_map = {}
backlog_layers = {"model_1": {}, "model_2": {}, }

os.makedirs(merged_model_folder, exist_ok=True)

for file_to_copy in [file for file in os.listdir(model_1_folder) if (file.endswith('.json') or file.endswith('.txt')) and not file.endswith('.index.json')]:
    shutil.copy(f'{model_1_folder}/{file_to_copy}', merged_model_folder)

for model_file_idx, model_file in enumerate(model_files):
    merged_model = {}

    model_file_idx += 1

    model_1_file, model_2_file = model_file

    model_1_layers = {}
    model_2_layers = {}

    params_size = 0

    if model_1_file is not None:
        print('[*] Reading', f"{model_1_folder}/{model_1_file}")
        if model_1_file.endswith('safetensors'):
            model_1_layers_sf = sft.safe_open(f"{model_1_folder}/{model_1_file}", framework="pt", device=torch_map_location)
            for sf_key in model_1_layers_sf.keys():
                model_1_layers[sf_key] = model_1_layers_sf.get_tensor(sf_key)
            del model_1_layers_sf
        else:
            model_1_layers = sft.torch.load(f"{model_1_folder}/{model_1_file}", map_location=torch_map_location)

    if model_2_file is not None:
        print('[*] Reading', f"{model_2_folder}/{model_2_file}")
        if model_2_file.endswith('safetensors'):
            model_2_layers_sf = sft.safe_open(f"{model_2_folder}/{model_2_file}", framework="pt", device=torch_map_location)
            for sf_key in model_2_layers_sf.keys():
                model_2_layers[sf_key] = model_2_layers_sf.get_tensor(sf_key)
            del model_2_layers_sf
        else:
            model_2_layers = sft.torch.load(f"{model_2_folder}/{model_2_file}", map_location=torch_map_location)

    model_1_layers.update(backlog_layers['model_1'])
    model_2_layers.update(backlog_layers['model_2'])

    backlog_layers['model_1'] = {}
    backlog_layers['model_2'] = {}

    for backlog_layer in set(model_1_layers).symmetric_difference(set(model_2_layers)):
        if backlog_layer in model_1_layers:
            backlog_layers['model_1'][backlog_layer] = model_1_layers[backlog_layer]
        if backlog_layer in model_2_layers:
            backlog_layers['model_2'][backlog_layer] = model_2_layers[backlog_layer]

    for common_layer in set(model_1_layers).intersection(set(model_2_layers)):
        model_1_layers[common_layer] = model_1_ratio * model_1_layers[common_layer]
        model_2_layers[common_layer] = model_2_ratio * model_2_layers[common_layer]

        merged_model[common_layer] = model_1_layers[common_layer] + model_2_layers[common_layer]

        del model_1_layers[common_layer]
        del model_2_layers[common_layer]

        bin_weight_map[common_layer] = f'{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}'

        params_size += sys.getsizeof(merged_model[common_layer].storage())

    print('[*] Memory used:', format_size(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), end='\n\n')

    del model_1_layers
    del model_2_layers

    gc.collect()

if len(model_files) > 1:
    print(f'[*] Saving {save_model_ext} weight map:', f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}.index.json')
    with open(f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}.index.json', 'w+') as f:
        f.write(json.dumps({"metadata": {"total_size": model_size_bytes}, "weight_map": bin_weight_map}, sort_keys=True, indent=4))

if (len(backlog_layers['model_1']) > 0 or len(backlog_layers['model_2'])):
    print('[WARN] Not all layers were merged, model might be in a broken state')
