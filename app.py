import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from source.Generate.generate import predict
from source.Function.search_Qdrant import  search_Article
from source.Function.utils import find_relevant_contexts
from source.Database.db_connection import DBConnection
load_dotenv()
SERVER_SSMS = os.getenv("SERVER_SSMS")
app = Flask(__name__)
db=DBConnection(SERVER_SSMS)
@app.route('/start-session', methods=['POST'])
def start_session():
    session_id=db.Create_Session()
    return jsonify({'session_id': session_id})
@app.route('/save-message', methods=['POST'])
def save_message():
    data = request.json
    session_id = data['session_id']
    sender = data['sender']
    message = data['message']
    db.Insert_Message(session_id,sender,message)
    return jsonify({'status': 'success', 'message': 'Message saved successfully'})
@app.route('/get-sessions', methods=['GET'])
def get_sessions():
    sessions=db.Get_Session()
    session_list = []
    for session in sessions:
        session_list.append({
            'id': session[0],
            'create_at': session[1],
            'first_message': session[2]
        })

    return jsonify({'sessions': session_list})
@app.route('/get-chat-history/<int:session_id>', methods=['GET'])
def get_chat_history(session_id):
    messages=db.Get_History(session_id)
    chat_history = []
    for message in messages:
        chat_history.append({
            'sender': message[0],
            'message': message[1],
            'send_at': message[2].strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify({'chat_history': chat_history})
@app.route('/delete-session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    db.Delete_Session(session_id)
    return jsonify({'status': 'success', 'message': 'Session deleted successfully'})
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['query']
    contexts, lst_Article_Quote = search_Article(user_input)
    answer_candidates = predict(contexts,user_input)
    lst_Relevant_Documents = find_relevant_contexts(lst_Article_Quote, answer_candidates)
    return jsonify({'answer': answer_candidates,'lst_Relevant_Documents': lst_Relevant_Documents})
if __name__ == '__main__':
    app.run(port=5000, debug=True)
