from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
class Gemini():
    def __init__(self,key_manager,model_gemini) :
        self.key_manager=key_manager
        self.model_gemini=model_gemini
    def query_generator(self,original_query: str) -> list[str]:
        query = original_query
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Bạn là một trợ lý hữu ích và có nhiệm vụ tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn gốc."),
                ("human", """Tạo chính xác 3 câu truy vấn tìm kiếm liên quan đến: {original_query}. Mỗi câu truy vấn trên một dòng mới. 
                Không được trả về nhiều hơn hoặc ít hơn 3 câu truy vấn. Đảm bảo không thêm bất kỳ văn bản nào khác ngoài 3 câu truy vấn này."""),
            ]
        )
        model = ChatGoogleGenerativeAI(
            google_api_key=self.key_manager.get_next_key(),
            model=self.model_gemini,
            temperature=0
        )
        query_generator_chain = (
            prompt | model | StrOutputParser()
        )
        result = query_generator_chain.invoke({"original_query": query})
        generated_queries = result.strip().split('\n')
        if len(generated_queries) > 3:
            generated_queries = generated_queries[:len(generated_queries) - 1]
        queries = [query] + generated_queries
        return queries
    def prompt_template(self,docs: List[Tuple], original_query: str) -> str:
        context = "\n".join([doc for doc in docs])
        response_prompt = ChatPromptTemplate.from_messages(
        [ ("system", "Bạn là một trợ lý chuyên gia về Pháp Luật Việt Nam. Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến Pháp Luật Việt Nam dựa trên thông tin đã được cung cấp."
            "Câu trả lời phải dựa trên thông tin được cung cấp và tuân thủ các yêu cầu trả lời sau."),

            ("human", f"""
                Dưới đây là câu hỏi cần bạn trả lời:
                '{original_query}'

                Bạn hãy trả lời câu hỏi này dựa trên nội dung sau:
                {context}

            Yêu cầu trả lời:
                1. Phân tích kỹ câu hỏi:
                - Hiểu đầy đủ ý nghĩa của câu hỏi, bao gồm cả các từ đồng nghĩa, cách diễn đạt tương tự hoặc biến thể ngữ nghĩa.
                - Bắt đầu câu trả lời bằng một câu mở đầu rõ ràng, liên quan trực tiếp đến ý chính trong câu hỏi. Ví dụ: Nếu câu hỏi là "Luật lao động là gì?" thì câu trả lời nên bắt đầu bằng: "Luật lao động là…".
                - Xác định các khía cạnh chính cần trả lời để tránh bỏ sót ý quan trọng.
                2. Trả lời câu hỏi:
                - Trình bày dưới dạng các đoạn văn logic, mỗi đoạn giải thích một khía cạnh hoặc ý chính của câu hỏi.
                - Chia câu trả lời thành các mục rõ ràng (ví dụ: 1, 2, 3), với mỗi mục phải liên quan trực tiếp đến tiêu đề hoặc ý chính được đề cập.
                3. Sử dụng thông tin cung cấp:
                - Trả lời dựa trên toàn bộ thông tin từ nội dung cung cấp. Không thêm thông tin ngoài nguồn cung cấp.
                4. Trường hợp không đủ dữ liệu:
                - Nếu không có thông tin liên quan, phản hồi: "Trong bộ dữ liệu không có thông tin."
                5. Đảm bảo chất lượng câu trả lời:
                - Câu trả lời cần đầy đủ, rõ ràng, dễ hiểu và có cấu trúc chặt chẽ.
                - Đảm bảo không để sót bất kỳ phần quan trọng nào, đồng thời tránh các thông tin dư thừa hoặc không liên quan.
                6. Phong cách trình bày:
                - Chuyên nghiệp, chính xác và phù hợp với ngữ cảnh Pháp Luật Việt Nam.
            """)
        ]
        )
        return response_prompt
    def generate_response(self,original_query: str, docs: List) -> str:
        response_model = ChatGoogleGenerativeAI(
            google_api_key=self.key_manager.get_next_key(),
            model=self.model_gemini,
            temperature=0.1,
            max_tokens=3000,
            top_p=0.6,
        )
        response_chain = self.prompt_template(docs, original_query) | response_model | StrOutputParser()
        final_response = response_chain.invoke({"original_query": original_query}).strip()
        return final_response 