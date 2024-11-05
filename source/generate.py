import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np
model = AutoModelForQuestionAnswering.from_pretrained("quanghuy123/fine-tuning-bert-for-QA",token='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')
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
MAX_LENGTH = 512
STRIDE = 320
N_BEST = 120
MAX_ANSWER_LENGTH = 2000
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
                            "logit_score": start_logits[start_index] + end_logits[end_index],
                        }
                        answers.append(answer)
        if answers:
            answers.sort(key=lambda x: x["logit_score"], reverse=True)
            answer_final.append(answers[0]["text"])
        else: 
            return "Không có câu trả lời"
    return answer_final  
 