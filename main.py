import Text_Content_Query.query_Generator_Gemini as qGG
import os

from API_Supplier.apikey_GEMINI import APIKeyManager

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, send_file
app = Flask(__name__)

API_KEYS = os.getenv('API_KEYS').split(',')
key_manager = APIKeyManager(API_KEYS)

@app.route('/test_query', methods=['POST'])
def post_query_test():
   
    data = request.get_json()
    user_query = data.get('user_query')

    response = qGG.get_documents(user_query, key_manager)
    
    return {
        'response': response
    }, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)