from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese", torch_dtype="auto", device_map="cuda:0")
toke