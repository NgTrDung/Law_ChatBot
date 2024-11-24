import os

from dotenv import load_dotenv
from source.apikeys_GEMINI import APIKeyManager
from flask import Flask, request, jsonify, render_template
from source.generate import predict, rerank_By_Cosin_BM25
from source.search_Qdrant import  search_Article
from source.utils import find_relevant_contexts
from source.T5Sum import summarize_text

load_dotenv()
APIS_GEMINI_LIST = os.getenv('APIS_GEMINI_LIST').split(',')
key_manager = APIKeyManager(APIS_GEMINI_LIST)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['query']

    contexts, lst_Article_Quote = search_Article(user_input, key_manager)
    answer_candidates = predict(contexts,user_input)
    answers_Generate = rerank_By_Cosin_BM25(user_input,answer_candidates)
    lst_Relevant_Documents = find_relevant_contexts(lst_Article_Quote, answers_Generate)

    answers_T5Sum = summarize_text(answers_Generate)
    print("Generated Answers:\n",answers_Generate,"\n\n")
    print("Relevant Documents:\n",lst_Relevant_Documents,"\n\n")
    print("Summarized Answers:\n",answers_T5Sum) 

    return jsonify({'answer': answers_T5Sum,'lst_Relevant_Documents': lst_Relevant_Documents})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
