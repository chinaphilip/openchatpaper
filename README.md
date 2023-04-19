# OpenChatPaper

![logo](./logo.png)

Forked from [Openchatpaper](https://github.com/liuyixin-louis/ChatPaper). An open-source version that attempts to reimplement [ChatPDF](https://www.chatpdf.com/). A different dialogue version of another [ChatPaper](https://github.com/chinaphilip/openchatpaper) project. 

本代码库是从[Openchatpaper](https://github.com/liuyixin-louis/ChatPaper)中fork过来。做了部分更改，试图实现基于开源对话模型的 [ChatPDF](https://www.chatpdf.com/) 的版本。
目前的对话模型使用的是[Vicuna-13b 8bit](https://github.com/lm-sys/FastChat)版本，文本检索模型使用的是["sentence-transformers/msmarco-distilbert-base-v4"](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4)，

**News**
- **Sat. Apr.1, 2023:** Add some buttons to get some basic aspects of paper quickly.

![image](https://user-images.githubusercontent.com/53036760/229304107-b3c38813-495e-4610-a6f4-379dfb8e2806.png)


## Setup

1. Install dependencies (tested on Python 3.9)

```bash
 pip install -r requirements.txt
```

2. Setup and lauch GROBID local server (add & at the end of command to run the program in the background)

```bash
bash serve_grobid.sh
```

3. Setup backend

```bash
python backend.py --port 5000 --host localhost
```

4. Frontend 

```bash
streamlit run frontend.py --server.port 8502 --server.address localhost
```

## 中文配置文档
> 程序在`Python>=3.9`, `Ubuntu 20.04`下测试，若在其他平台测试出错，欢迎提issue

1. 创建一个Python的环境（推荐使用anaconda，关于如何安装请查阅[其他教程](https://zhuanlan.zhihu.com/p/123188004)），创建环境后激活并且安装依赖
```bash
 conda create -n cpr python=3.9
 conda activate cpr
 pip install -r requirements.txt
```

2. 确保本机安装了java环境，如果`java -version`成功放回版本即说明安装成功。关于如何安装JAVA请查阅[其他教程](https://www.runoob.com/java/java-environment-setup.html)

3. [GROBID](https://github.com/kermitt2/grobid)是一个开源的PDF解析器，我们会在本地启动它用来解析输入的pdf。执行以下命令来下载GROBID和运行，成功后会显示`EXECUTING[XXs]`

```bash
bash serve_grobid.sh
```

![image](https://user-images.githubusercontent.com/53036760/229299669-7425c18d-c0fe-4e53-8022-5cd094c5c0cf.png)

3. 开启后端进程：每个用户的QA记录放进一个缓存pool里

```bash
python backend.py --port 5000 --host localhost
```

4. 最后一步，开启Streamlit前端，访问`http://localhost:8502`，在API处输入OpenAI的APIkey（[如何申请?](https://juejin.cn/post/7203009064719400997)），上传PDF文件解析完成后便可开始对话

```bash
streamlit run frontend.py --server.port 8502 --server.address localhost
```

## Demo Example

- Prepare an [OpenAI API key](https://platform.openai.com/account/api-keys) and then upload a PDF to start chatting with the paper. 

![image-20230318232056584](https://s2.loli.net/2023/03/19/SbsuLQJpdqePoZV.png)

## Implementation Details

- Greedy Dynamic Context: Since the max token limit, we select the most relevant paragraphs in the pdf for each user query. Our model split the text input and output by the chatbot into four part: system_prompt (S), dynamic_source (D), user_query (Q), and model_answer(A). So upon each query, we first rank all the paragraphs by using a sentence_embedding model to calculate the similarity distance between the query embedding and all source embeddings. Then we compose the dynamic_source using a greedy method by to gradually push all relevant paragraphs (maintaing D <= MAX_TOKEN_LIMIT - Q - S - A - SOME_OVERHEAD). 

- Context Truncating: When context is too long, we now we simply pop out the first QA-pair. 

## TODO

- [ ] **Context Condense**: how to deal with long context? maybe we can tune a soft prompt to condense the context
- [ ] **Poping context out based on similarity**
- [ ] **Handling paper with longer pages**

## Cooperation & Contributions

Feel free to reach out for possible cooperations or Contributions! (aauiuui@163.com)

## References

1. SciPDF Parser: https://github.com/titipata/scipdf_parser 
2. St-chat: https://github.com/AI-Yash/st-chat
3. Sentence-transformers: https://github.com/UKPLab/sentence-transformers
4. ChatGPT Chatbot Wrapper: https://github.com/acheong08/ChatGPT
5. Openchatpaper https://github.com/liuyixin-louis/ChatPaper
6. langchain-ChatGLM  https://github.com/imClumsyPanda/langchain-ChatGLM
7. Vicuna https://github.com/lm-sys/FastChat



