from flask import jsonify
from embedding_model import SentenceTransformerMsmarcoModel

from pdf_parser import GrobidSciPDFPaser
from chatbot import VicunaChatbot
from chat_pdf import ChatPDF
from config import Default_model_name,DEFAULT_max_new_tokens, DEFAULT_DEVICE,MAX_TOKEN_MODEL_MAP, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_PRESENCE_PENALTY, DEFAULT_FREQUENCY_PENALTY, DEFAULT_REPLY_COUNT
from flask import Flask, request
from base_class import Pool
app = Flask(__name__)
chatpdf_pool = Pool()

embedding_model = SentenceTransformerMsmarcoModel()
bot = VicunaChatbot(
    model_name=Default_model_name,
    max_new_tokens=DEFAULT_max_new_tokens,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    presence_penalty=DEFAULT_PRESENCE_PENALTY,
    frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
    reply_count=DEFAULT_REPLY_COUNT,
    device=DEFAULT_DEVICE
)
print("对话模型加载完毕")



@app.route("/query/", methods=['POST', 'GET'])
def query():
    #api_key = request.headers.get('Api-Key')
    try:
        pdf_link = request.json['pdf_link']
        user_stamp = request.json['user_stamp']
        user_query = request.json['user_query']
    except:
        print("参数解析错误")
    print(
        #"api_key", api_key,
        "pdf_link", pdf_link,
        "user_stamp", user_stamp,
        "user_query", user_query
    )
    chat_pdf = None
    if user_stamp not in chatpdf_pool:
        print(f"User {user_stamp} not in pool, creating new chatpdf")
        pdf = GrobidSciPDFPaser(
            pdf_link=pdf_link
        )
        print("pdf解析完毕")
        # Initialize the ChatPDF
        chat_pdf = ChatPDF(
            pdf=pdf,
            bot=bot,
            embedding_model=embedding_model,
            user_stamp=user_stamp
        )
        print("creating chatpdf successfully")
        chatpdf_pool[user_stamp] = chat_pdf
    else:
        print("user_stamp", user_stamp, "already exists")
        chat_pdf = chatpdf_pool[user_stamp]

    try:
        response = chat_pdf.chat(user_query)
        code = 200
        json_dict = {
            "code": code,
            "response": response
        }
    except Exception as e:
        print(response)
        code = 500
        json_dict = {
            "code": code,
            "response": str(e)
        }
    return jsonify(json_dict)


@app.route("/", methods=['GET'])
def index():
    return "Hello World!"


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
