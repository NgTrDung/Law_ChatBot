import re
import os

from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline

T5SUM_MODEL_TOKENIZER = os.getenv('T5SUM_MODEL_TOKENIZER')
T5SUM_MODEL_GENERATE = os.getenv('T5SUM_MODEL_GENERATE')
tokenizer = AutoTokenizer.from_pretrained(T5SUM_MODEL_TOKENIZER, trust_remote_code=True)
model = T5ForConditionalGeneration.from_pretrained(T5SUM_MODEL_GENERATE)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token  
tokenizer.add_eos_token = True
def summarize_text(text, max_chunk_length=512, stride=256, max_summary_length=1024, num_beams=5, repetition_penalty=2.0):
    def preprocess_text(text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s*\.{2,}\s*', '. ', text)  
        text = re.sub(r'\s*\.\s*', '. ', text)     
        text = text.strip(" .")
        return text
    def split_text_into_chunks(text, max_length=512, stride=256):
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) == 0:
                break
            chunks.append(chunk)
        return chunks
    text = preprocess_text(text)
    pipe = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        num_beams=num_beams,                
        repetition_penalty=repetition_penalty,
    )
    chunks = split_text_into_chunks(text, max_length=max_chunk_length, stride=stride)
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        result = pipe(chunk_text, max_length=max_summary_length, truncation=True)
        summaries.append(result[0]['summary_text'])
    final_summary = " ".join(summaries)
    return final_summary
