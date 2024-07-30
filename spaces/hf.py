from huggingface_hub import login, snapshot_download
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

login(token="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")

models = ["internlm/internlm2_5-7b-chat"]

for model in models:
    try:
        snapshot_download(repo_id=model,local_dir="./models")
    except Exception as e:
        print(e)
        pass