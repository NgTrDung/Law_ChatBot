import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from source.Gemini.apikeys_gemini import APIKeyManager
from source.Gemini.gemini import Gemini
from dotenv import load_dotenv
import os 
load_dotenv()
MODEL_EXTRACT=os.getenv("MODEL_EXTRACT")
TOKENIZER=os.getenv("TOKENIZER")
APIS_GEMINI_LIST = os.getenv('APIS_GEMINI_LIST').split(',')
key_manager = APIKeyManager(APIS_GEMINI_LIST)
MODEL_GEMINI = os.getenv("MODEL_GEMIMI")
model =AutoModelForQuestionAnswering.from_pretrained(MODEL_EXTRACT, token='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')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
MAX_LENGTH = 512
STRIDE = 380
N_BEST = 180
MAX_ANSWER_LENGTH = 2000
model_gemini=Gemini(key_manager,MODEL_GEMINI)
def predict(contexts, question):
    answer_final = []
    for context in contexts:
        inputs = tokenizer(
            question,
            context,
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
        start_logits = outputs.start_logits.squeeze().cpu().numpy()
        end_logits = outputs.end_logits.squeeze().cpu().numpy()
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        answers = []
        start_indexes = np.argsort(start_logits)[-N_BEST:][::-1].tolist()
        end_indexes = np.argsort(end_logits)[-N_BEST:][::-1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                if end_index < start_index or end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                    continue
                if offsets[start_index][0] is not None and offsets[end_index][1] is not None:
                    answer_text = context[offsets[start_index][0]: offsets[end_index][1]].strip()
                    if answer_text:
                        answer = {
                            "text": answer_text,
                            "score": start_logits[start_index] + end_logits[end_index],
                        }
                        answers.append(answer)
        if answers:
            answers.sort(key=lambda x: x["score"], reverse=True)
            best_answer = answers[0]['text']
            if best_answer not in answer_final:
                answer_final.append(best_answer)
                print(best_answer)
        else: 
            return "Không có câu trả lời"
        result_final=model_gemini.generate_response(question,answer_final)
    return result_final 

