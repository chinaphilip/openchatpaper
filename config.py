

MAX_TOKEN_MODEL_MAP = {
    "gpt-3.5-turbo": 4096,
}

PDF_SAVE_DIR = "./files/"


Default_model_name="E:/vicuna_converted"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1
DEFAULT_PRESENCE_PENALTY = 0
DEFAULT_FREQUENCY_PENALTY = 0
DEFAULT_REPLY_COUNT = 1
DEFAULT_DEVICE="cuda"#type=str, choices=["cpu", "cuda", "mps"],)
DEFAULT_num_gpus=1
DEFAULT_load_8bit=True
DEFAULT_debug=False
DEFAULT_max_new_tokens=250

quickbuttons = {
    "summarize": "Please summarize this paper.",
    "contribution": "What is the contribution of this paper?",
    "novelty": "What is the novelty of this paper?",
    "strength": "What are the strengths of this paper?",
    "drawback": "What are the drawbacks of this paper?",
    "improvement": "What might be the improvements of this paper?",
}