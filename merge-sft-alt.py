import os
import gc
import json
import shutil
import resource

import safetensors.torch as sft

from itertools import zip_longest

secondary_model_ratio = 0.75

base_model_folder = 'OPT-350M-Erebus'
secondary_model_folder = 'OPT-350M-Nerys-v2-100MB'

merged_model_folder = 'OPT-350M-Nereberys-pt'

torch_map_location = 'cpu'

save_type = 'pytorch'

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

print(f"[*] Merging models\n\t{base_model_folder.split('/')[-1]}\n\t{secondary_model_folder.split('/')[-1]}\n")


def format_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


base_model_files = [file for file in os.listdir(base_model_folder) if file.endswith('.bin') or file.endswith('.safetensors')]
secondary_model_files = [file for file in os.listdir(secondary_model_folder) if file.endswith('.bin') or file.endswith('.safetensors')]

base_model_files.sort()
secondary_model_files.sort()

max_files_length = len(base_model_files) if len(base_model_files) > len(secondary_model_files) else len(secondary_model_files)

model_files = [file for file in zip_longest(base_model_files, secondary_model_files)]

model_size_bytes = 0

for file in base_model_files:
    model_size_bytes += os.path.getsize(f'{base_model_folder}/{file}')

bin_weight_map = {}
backlog_layers = {"base_model": {}, "secondary_model": {}, }

folder_created = False

for model_file_idx, model_file in enumerate(model_files):
    merged_model = {}

    model_file_idx += 1

    base_model_file, secondary_model_file = model_file

    base_model_layers = {}
    secondary_model_layers = {}

    if base_model_file is not None:
        print('[*] Reading', f"{base_model_folder}/{base_model_file}")
        if base_model_file.endswith('safetensors'):
            base_model_layers_sf = sft.safe_open(f"{base_model_folder}/{base_model_file}", framework="pt", device=torch_map_location)
            for sf_key in base_model_layers_sf.keys():
                base_model_layers[sf_key] = base_model_layers_sf.get_tensor(sf_key)
            del base_model_layers_sf
        else:
            base_model_layers = sft.torch.load(f"{base_model_folder}/{base_model_file}", map_location=torch_map_location)

    if secondary_model_file is not None:
        print('[*] Reading', f"{secondary_model_folder}/{secondary_model_file}")
        if secondary_model_file.endswith('safetensors'):
            secondary_model_layers_sf = sft.safe_open(f"{secondary_model_folder}/{secondary_model_file}", framework="pt", device=torch_map_location)
            for sf_key in secondary_model_layers_sf.keys():
                secondary_model_layers[sf_key] = secondary_model_layers_sf.get_tensor(sf_key)
            del secondary_model_layers_sf
        else:
            secondary_model_layers = sft.torch.load(f"{secondary_model_folder}/{secondary_model_file}", map_location=torch_map_location)

    base_model_layers.update(backlog_layers['base_model'])
    secondary_model_layers.update(backlog_layers['secondary_model'])

    backlog_layers['base_model'] = {}
    backlog_layers['secondary_model'] = {}

    for backlog_layer in set(base_model_layers).symmetric_difference(set(secondary_model_layers)):
        if backlog_layer in base_model_layers:
            backlog_layers['base_model'][backlog_layer] = base_model_layers[backlog_layer]
        if backlog_layer in secondary_model_layers:
            backlog_layers['secondary_model'][backlog_layer] = secondary_model_layers[backlog_layer]

    for common_layer in set(base_model_layers).intersection(set(secondary_model_layers)):
        w_base_model = (1 - secondary_model_ratio) * base_model_layers[common_layer]
        w_secondary_model = secondary_model_ratio * secondary_model_layers[common_layer]

        if common_layer.startswith('model.decoder.layers') and int(common_layer.split('.')[3]) % 2 == 0:
            merged_model[common_layer] = (w_base_model * base_model_layers[common_layer]) + (w_secondary_model * secondary_model_layers[common_layer])
        else:
            merged_model[common_layer] = base_model_layers[common_layer]

        bin_weight_map[common_layer] = f'{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}'

    if not folder_created:
        os.makedirs(merged_model_folder, exist_ok=True)

        for file_to_copy in [file for file in os.listdir(base_model_folder) if (file.endswith('.json') or file.endswith('.txt')) and not file.endswith('.index.json')]:
            shutil.copy(f'{base_model_folder}/{file_to_copy}', merged_model_folder)

    folder_created = True

    if len(model_files) == 1:
        print(f'[*] Saving model: {merged_model_folder}/{save_model_prefix}.{save_model_ext}')
        if save_type == 'safetensors':
            sft.save_file(merged_model, f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}', metadata={'format': 'pt'})
        else:
            sft.torch.save(merged_model, f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}')
    else:
        print(f'[*] Saving shard: {merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}')
        if save_type == 'safetensors':
            sft.save_file(merged_model, f'{merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}', metadata={'format': 'pt'})
        else:
            sft.torch.save(merged_model, f'{merged_model_folder}/{save_model_prefix}-{(model_file_idx):05}-of-{max_files_length:05}.{save_model_ext}')

    print('[*] Memory used:', format_size(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), end='\n\n')

    del secondary_model_layers
    gc.collect()

if len(model_files) > 1:
    print(f'[*] Saving {save_model_ext} weight map:', f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}.index.json')
    with open(f'{merged_model_folder}/{save_model_prefix}.{save_model_ext}.index.json', 'w+') as f:
        f.write(json.dumps({"metadata": {"total_size": model_size_bytes}, "weight_map": bin_weight_map}, sort_keys=True, indent=4))

if (len(backlog_layers['base_model']) > 0 or len(backlog_layers['secondary_model'])):
    print('[WARN] Not all layers were merged, model might be in a broken state')
