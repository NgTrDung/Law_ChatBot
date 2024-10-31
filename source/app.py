import os
import search_Qdrant as s_Q

# from flask import Flask, request, jsonify,render_template
# from generate import predict_answer
from apikeys_GEMINI import APIKeyManager

APIS_GEMINI_LIST = os.getenv('APIS_GEMINI_LIST').split(',')
key_manager = APIKeyManager(APIS_GEMINI_LIST)

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/chatbot', methods=['POST'])

# def chatbot():
#     user_input = request.json['query']
    
#     contexts= Search_Context(user_input)
#     print(contexts)
#     answer= predict_answer(contexts,user_input)
#     print(answer)
#     return jsonify({'answer': answer})

def search_Article_Section(user_Query, key_manager):
    article_Section_Content_Results = s_Q.get_Article_Section_Content_Result(user_Query, key_manager)
    
    return article_Section_Content_Results

def search_Article(user_Query, key_manager):
    article_Document_Results = s_Q.get_Article_Content_Results(user_Query, key_manager)
    
    return article_Document_Results

if __name__ == '__main__':
    # app.run(debug=True)

    user_Query = "Thông điệp dữ liệu như file có cần công chứng hoặc chứng thực gì không để được xem là có giá trị như một văn bản, theo quy định của Luật?"
    results = search_Article(user_Query, key_manager)

    # In ra để xem ở cmd
    i=1
    for idx in results:
        print(f"Tài liệu {i}:")
        print(idx,"\n------------------------")
        i+=1

