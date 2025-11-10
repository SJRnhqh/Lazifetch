# src/lazifetch/utils/llm.py
# 导入外部库
import os


# 导入内部库
from ..model import OpenAILLM


def get_llms():
    if "MAIN_LLM_MODEL" not in os.environ or os.environ["MAIN_LLM_MODEL"] == "":
        raise ValueError("MAIN_LLM_MODEL is not set")
    if "CHEAP_LLM_MODEL" not in os.environ or os.environ["CHEAP_LLM_MODEL"] == "":
        raise ValueError("CHEAP_LLM_MODEL is not set")
    main_llm = os.environ.get("MAIN_LLM_MODEL","qwen3-235b-a22b-thinking-2507")
    cheap_llm= os.environ.get("CHEAP_LLM_MODEL","qwen3-235b-a22b-thinking-2507")
    main_llm = OpenAILLM(model=main_llm)
    cheap_llm = OpenAILLM(model=cheap_llm)
    return main_llm,cheap_llm