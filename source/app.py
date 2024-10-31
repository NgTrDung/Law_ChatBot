import os

from flask import Flask, request, jsonify,render_template
from generate import predict
from search_Qdrant import search_Article_Section,search_Article
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])

def chatbot():
    user_input = request.json['query']
    
    contexts= search_Article(user_input)
    print(contexts)
    answer= predict(contexts,user_input)
    print(answer)
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)


