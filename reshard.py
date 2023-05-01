from transformers import AutoModelForCausalLM

model_path = 'OPT-350M-Nerys-v2'
shard_size = '100MB'

model = AutoModelForCausalLM.from_pretrained(model_path)
model.save_pretrained(f'{model_path}-{shard_size}', max_shard_size=shard_size)
