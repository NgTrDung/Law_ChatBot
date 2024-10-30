import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
model_path="../model_2/"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
def predict_answer(contexts, question, max_length=512, stride=200):
    answers=[]
    # Tokenize câu hỏi và context
    for context in contexts:
        inputs = tokenizer(
            question,
            context,
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_tensors="pt",
            padding="max_length",
        )

        # Dự đoán bằng mô hình đã fine-tuned
        with torch.no_grad():
            outputs = model(**inputs)

        # Lấy giá trị logits cho start và end position
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Tìm các index tốt nhất cho start và end position
        start_index = torch.argmax(start_logits, dim=-1).item()
        end_index = torch.argmax(end_logits, dim=-1).item()

        # Giải mã câu trả lời từ context
        answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1], skip_special_tokens=True)
        answers.append(answer)
    return answers