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

#-----------------------------------------------------------#



import pyodbc  # Thư viện để kết nối với SQL Server

# Hàm kết nối với database SQL Server
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=DESKTOP-B86U75E;'
        'DATABASE=Law_ChatBot_DB;'
        'Trusted_Connection=yes;'
    )

    return conn

@app.route('/start-session', methods=['POST'])
def start_session():
    # Kết nối tới database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Thêm bản ghi mới vào bảng Chat_Sessions
    cursor.execute("INSERT INTO Chat_Sessions (create_at) OUTPUT INSERTED.id VALUES (GETDATE())")
    session_id = cursor.fetchone()[0]  # Lấy ID của phiên vừa tạo
    
    conn.commit()  # Xác nhận thay đổi
    conn.close()  # Đóng kết nối
    
    # Trả về ID của session
    return jsonify({'session_id': session_id})

@app.route('/save-message', methods=['POST'])
def save_message():
    # Lấy dữ liệu từ yêu cầu POST
    data = request.json
    session_id = data['session_id']
    sender = data['sender']
    message = data['message']

    # Kết nối tới database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Thêm tin nhắn vào bảng Chat_Messages
    cursor.execute(
        """
        INSERT INTO Chat_Messages (session_id, sender, message, send_at)
        VALUES (?, ?, ?, GETDATE())
        """,
        (session_id, sender, message)
    )
    conn.commit()  # Xác nhận thay đổi
    conn.close()  # Đóng kết nối

    return jsonify({'status': 'success', 'message': 'Message saved successfully'})

@app.route('/get-sessions', methods=['GET'])
def get_sessions():
    # Kết nối tới database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Lấy danh sách các phiên chat có ít nhất một tin nhắn từ user
    query = """
    SELECT 
        cs.id, 
        cs.create_at, 
        (SELECT TOP 1 message FROM Chat_Messages 
         WHERE session_id = cs.id AND sender = 'user' 
         ORDER BY send_at ASC) AS first_message
    FROM Chat_Sessions cs
    WHERE EXISTS (
        SELECT 1 FROM Chat_Messages 
        WHERE session_id = cs.id AND sender = 'user'
    )
    ORDER BY cs.create_at DESC
    """
    cursor.execute(query)
    sessions = cursor.fetchall()

    conn.close()

    # Chuyển đổi dữ liệu thành danh sách JSON
    session_list = []
    for session in sessions:
        session_list.append({
            'id': session[0],
            'create_at': session[1],
            'first_message': session[2]
        })

    return jsonify({'sessions': session_list})



#-----------------------------------------------------------#

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
