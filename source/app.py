from flask import Flask, request, jsonify,render_template
from search_Qdrant import Search_Context
from generate import predict_answer
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])

def chatbot():
    user_input = request.json['query']
    
    contexts= Search_Context(user_input)
    print(contexts)
    answer= predict_answer(contexts,user_input)
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)