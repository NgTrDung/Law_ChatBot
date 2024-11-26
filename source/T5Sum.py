import re

from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline

def summarize_text(text, max_chunk_length=512, stride=256, max_summary_length=1024, num_beams=5, repetition_penalty=2.0):
    # Load model và tokenizer
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base', trust_remote_code=True)
    model = T5ForConditionalGeneration.from_pretrained("hghaan/t5_summary")
    
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.add_eos_token = True
    
    # Hàm tiền xử lý văn bản
    def preprocess_text(text):
        # Xóa dấu xuống dòng và các khoảng trắng thừa
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Loại bỏ dấu chấm đơn lẻ và dấu chấm thừa
        text = re.sub(r'\s*\.{2,}\s*', '. ', text)  # Xóa dấu chấm liên tiếp
        text = re.sub(r'\s*\.\s*', '. ', text)       # Xóa khoảng trắng xung quanh dấu chấm
        
        # Loại bỏ dấu chấm ở đầu và cuối chuỗi nếu có
        text = text.strip(" .")
        
        return text
    
    # Hàm chia văn bản thành các đoạn nhỏ
    def split_text_into_chunks(text, max_length=512, stride=256):
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) == 0:
                break
            chunks.append(chunk)
        return chunks
    
    # Tiền xử lý văn bản
    text = preprocess_text(text)
    
    # Tạo pipeline tóm tắt
    pipe = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        num_beams=num_beams,                
        repetition_penalty=repetition_penalty,
    )
    
    # Chia văn bản thành các đoạn nhỏ
    chunks = split_text_into_chunks(text, max_length=max_chunk_length, stride=stride)
    summaries = []
    
    for chunk in chunks:
        # Giải mã các token chunk thành văn bản
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        
        # Tóm tắt từng đoạn nhỏ bằng pipeline
        result = pipe(chunk_text, max_length=max_summary_length, truncation=True)
        
        # Thêm văn bản tóm tắt vào danh sách summaries
        summaries.append(result[0]['summary_text'])
    
    # Ghép các tóm tắt của từng đoạn thành tóm tắt cuối cùng
    final_summary = " ".join(summaries)
    return final_summary
