def response_failed(content:list=[], message:str=""):
    result = {
            "success": False, 
            "message": message, 
            "answers": content
        }
    return result

def response_success(content:list, message:str="Đây là câu trả lời cho câu hỏi của bạn!"):
    result = {
            "success": True, 
            "message": message, 
            "answers": content
        }
    return result
