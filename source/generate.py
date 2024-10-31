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
    # Token hóa ngữ cảnh và câu hỏi
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

        # Dự đoán với mô hình
        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})

        # Lấy logits cho vị trí bắt đầu và kết thúc
        start_logits = outputs.start_logits.squeeze().cpu().numpy()
        end_logits = outputs.end_logits.squeeze().cpu().numpy()

        # Tạo danh sách các offsets từ tokenization
        offsets = inputs["offset_mapping"][0].cpu().numpy()

        # Duyệt qua các chỉ số và tìm các câu trả lời
        answers = []
        start_indexes = np.argsort(start_logits)[-N_BEST:][::-1].tolist()
        end_indexes = np.argsort(end_logits)[-N_BEST:][::-1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # Kiểm tra tính hợp lệ của start và end index
                if end_index < start_index or end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                    continue
                
                # Lấy câu trả lời từ offsets
                if offsets[start_index][0] is not None and offsets[end_index][1] is not None:
                    answer_text = context[offsets[start_index][0]: offsets[end_index][1]].strip()
                    if answer_text:
                        answer = {
                            "text": answer_text,
                            "logit_score": start_logits[start_index] + end_logits[end_index],
                        }
                        answers.append(answer)

        if answers:
            # Sắp xếp câu trả lời theo điểm số
            answers.sort(key=lambda x: x["logit_score"], reverse=True)
            answer_final.append(answers[0]["text"])
        else: 
            return "Không có câu trả lời"
    return answer_final  # Trả về danh sách các câu trả lời
 