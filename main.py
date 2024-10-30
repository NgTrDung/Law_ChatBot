import os
import module_RAG.search_Qdrant as s_Q

from dotenv import load_dotenv
from module_RAG.apikeys_GEMINI import APIKeyManager

APIS_GEMINI_LIST = os.getenv('APIS_GEMINI_LIST').split(',')
key_manager = APIKeyManager(APIS_GEMINI_LIST)

def handle_Query_HandBook(user_Query):
    article_Results = s_Q.handle_Query(user_Query, key_manager)
    idx = 1
    lst_Article_Content = []

    for doc, score in article_Results:
        print("Document:",idx)
        print(doc.metadata["combine_Article_Content"])
        lst_Article_Content.append(doc.metadata["combine_Article_Content"])
        print("-----------------------\n")
        idx += 1

    return lst_Article_Content
    
if __name__ == "__main__":
    user_Query = "Luật quy định như thế nào về việc thành viên trong tổ hợp tác hoặc hợp tác xã sử dụng quyền hạn cho mục đích cá nhân? Có điều khoản nào ngăn cấm việc này không?"
    handle_Query_HandBook(user_Query)