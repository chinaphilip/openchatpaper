"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model ~/model_weights/llama-7b
"""
import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,LlamaTokenizer

from fastchat.conversation import conv_templates, SeparatorStyle
from fastchat.serve.compression import compress_module
from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from fastchat.serve.serve_chatglm import chatglm_generate_stream


def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.to("cuda")
    elif device == "mps":
        model.to("mps")

    if (device == "mps" or device == "cpu") and load_8bit:
        compress_module(model)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]#他这里是把prompt过长的长度部分给截断了，效果还居然没有太大影响，奇了怪了

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)#生成第二个token的时候就有attention mask了
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)#生成第二个token的时候就不用把前面的token再输入一遍了
            logits = out.logits
            past_key_values = out.past_key_values
        #这里看这个logits是所有token都会算的
        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")
        
        #下面这个是用来确定输出的token的
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))    

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False
        
        #下面开始解码，把id转为token 
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values


def main(args):
    model_name = args.model_name

    # Model
    model, tokenizer = load_model(args.model_name, args.device,
        args.num_gpus, args.load_8bit, args.debug)
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset:]
            generate_stream_func = chatglm_generate_stream
            skip_echo_len = len(conv.messages[-2][1]) + 1
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()
            skip_echo_len = len(prompt) + 1

        params = {
            "model": model_name,
            "prompt": prompt,#输入的消息包含在prompt里面了
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        print(f"{conv.roles[1]}: ", end="", flush=True)
        pre = 0
        
        for outputs in generate_stream_func(model, tokenizer, params, args.device):
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            print("产生了一次output")
            if now - 1 > pre:#这个好像是一个debug的代码，如果有一次输出了两个字符，就把这几乎话打印出来
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        #其实上面这整个for循环都没有什么意义，只需要让他一直generte就好了
        print(" ".join(outputs[pre:]), flush=True)

        conv.messages[-1][-1] = " ".join(outputs)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

if __name__=="main":
    args={}
    args["model_name"]="E:/vicuna_converted"
    args["device"]="cuda"#, type=str, choices=["cpu", "cuda", "mps"],)
    args["num_gpus"]="1"
    args["load_8bit"]="true"#"Use 8-bit quantization.")
    args["conv_template"]="v1"#, type=str, default="v1",help="Conversation prompt template.")
    args["temperature"]=0.7
    args["max_new_tokens"]=512
    args["debug"]="False"

    class DictToObject(object):
        def __init__(self, d):
            self.__dict__.update(d)

    args=DictToObject(args)
