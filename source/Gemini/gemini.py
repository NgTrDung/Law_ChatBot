from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple

class Gemini():
    def __init__(self,key_manager,model_gemini) :
        self.key_manager=key_manager
        self.model_gemini=model_gemini

    def query_generator(self, original_query: str) -> list[str]:
        query = original_query
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """
                   Bạn là một trợ lý AI chuyên gia về pháp luật Việt Nam, chuyên phân tích ngữ nghĩa và tối ưu hóa truy vấn tìm kiếm. 
                   Nhiệm vụ:
                    - Tạo 5 câu truy vấn tìm kiếm khác nhau dựa trên truy vấn gốc, giữ nguyên ý chính nhưng thay đổi cách diễn đạt.
                    - Tăng tính bao quát và chi tiết bằng cách sử dụng từ khóa liên quan và cụm từ pháp lý thông dụng.
                    - Tối ưu hóa phù hợp với tìm kiếm trên vector database.
                """),
                ("human", f"""
                Từ câu truy vấn gốc '"{original_query}"', hãy làm 3 yêu cầu bên dưới:
                ***Yêu cầu 1 Cách tạo sinh :***
                -Tạo sinh câu hỏi bằng cách đọc kĩ hiểu câu hỏi sau đó phân loại truy vấn theo hai nhóm để biết cách tạo sinh các câu hỏi cho phù:
                    1. Người dùng không biết luật (tìm kiếm thông tin cơ bản).
                    2. Người dùng biết luật (tìm kiếm thông tin chi tiết, cụ thể với điều luật hoặc tài liệu pháp lý)
                Ví dụ cho người dùng không biết luật thì sẽ đổi thành:
                1."- 'Điều kiện để thành lập doanh nghiệp tại Việt Nam là gì?' có thể đổi thành 'Thành lập doanh nghiệp tại Việt Nam cần những điều kiện gì?'"
                2."- 'Quyền lợi của người lao động khi tham gia bảo hiểm xã hội là gì?' có thể đổi thành 'Bảo hiểm xã hội mang lại quyền lợi gì cho người lao động?'"
                3."- 'Các bước thủ tục ly hôn theo quy định của pháp luật Việt Nam như thế nào?' có thể đổi thành 'Thủ tục ly hôn theo pháp luật Việt Nam bao gồm những bước nào?'"
                4.- 'Tôi có thể khởi kiện vì bị xâm phạm quyền lợi như thế nào?' có thể đổi thành 'Cách khởi kiện khi quyền lợi bị xâm phạm là gì?'"
                5."- 'Quyền lợi của người lao động khi bị sa thải theo quy định của pháp luật là gì?' có thể đổi thành 'Khi bị sa thải, người lao động có quyền lợi gì theo pháp luật Việt Nam?'"

                Ví dụ cho người dùng biết luật thì sẽ đổi thành:
                -Khi người dùng biết luật
                "- 'Điều 12 của Luật Doanh nghiệp số 68/2014/QH13 quy định như thế nào về việc thành lập doanh nghiệp?'" " có thể đổi thành 'Luật Doanh nghiệp 68/2014/QH13 quy định về điều kiện thành lập doanh nghiệp tại Điều 12 như thế nào?'"
                "- 'Theo Luật Lao động số 45/2019/QH14, quyền lợi của người lao động khi tham gia bảo hiểm xã hội được quy định tại Chương III?'" " có thể đổi thành 'Quyền lợi của người lao động khi tham gia bảo hiểm xã hội theo Luật Lao động số 45/2019/QH14 được quy định ở đâu?'"
                "- 'Công ty tôi có thể tham gia vào hợp đồng lao động theo Điều 16 của Bộ luật Lao động 2012 không?'" " có thể đổi thành 'Bộ luật Lao động 2012 có quy định gì về hợp đồng lao động tại Điều 16 không?'"
                "- 'Thủ tục ly hôn theo pháp luật Việt Nam được quy định tại Điều 51 của Luật Hôn nhân và Gia đình 2014 như thế nào?'"" có thể đổi thành 'Luật Hôn nhân và Gia đình 2014 quy định thủ tục ly hôn ở Điều 51 ra sao?'"
                "- 'Điều 10 của Luật Đầu tư 2020 quy định về đầu tư vào ngành nghề nào tại Việt Nam?'" " có thể đổi thành 'Luật Đầu tư 2020 quy định ngành nghề nào được ưu tiên đầu tư tại Điều 10?'
                
                ***Yêu cầu 2:Cấu trúc***:
                - Các câu truy vấn không được lặp lại ý hoàn toàn.
                - Phải giữ cấu trúc rõ ràng, từ ngữ dễ hiểu, và có tính ứng dụng thực tiễn cao.
                - Dùng ngữ cảnh cụ thể hơn hoặc từ khóa liên quan nếu phù hợp.
                
                *** Yêu cầu 3:Số lượng *** 
                - Tạo chính xác 5 câu truy vấn tối ưu hóa nhất cho truy vấn gốc.
                - Chỉ trả về 5 câu truy vấn, không giải thích thêm.""")
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

        if len(generated_queries) > 5:
            generated_queries = generated_queries[:5]

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
        ])

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