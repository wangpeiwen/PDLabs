# coding=UTF-8
import torch
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    torch.cuda.memory._record_memory_history(max_entries=100000)
    model_name = "/home/kaiyuan/models/Qwen2.5-7B-Instruct"  # 替换为已下载的模型地址
    llm = LLM(model=model_name, dtype='float16')
    n = 16
    # 准备输入提示
    prompts = [
    "Hello, I'm kaiyuan",
    "Do you subscribe InfraTech?",
    ]

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,  # 控制生成文本的随机性，值越高越随机
        top_p=0.95,       # 控制采样范围，值越高生成文本越多样化
        max_tokens=50,     # 生成的最大 token 数量
        n=n
    )
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.memory._dump_snapshot("vllm_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)