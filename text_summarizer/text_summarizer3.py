# 3rd version of summarizer that allows us to define the schema of json output we want along with data validation

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

class Summarize(BaseModel):
  role: str = Field(default='Expert', description="Extract the role from the user question.")
  topic: str = Field(description="extract the topic to explain from the user question")
  lines: int = Field(gt = 1, default=4, description="Extract number of lines the user want the answer in.")

parser = PydanticOutputParser(pydantic_object=Summarize)

template1 = PromptTemplate(
  template="""Take the user input: {user_input} and extract the role you have to play, the topic to explain, and the number of lines.\n {format_instruction}""",
  input_variables=['user_input'],
  partial_variables={'format_instruction': parser.get_format_instructions()}
)

template2 = PromptTemplate(
  template="""You are a helpful {role}. Answer the user question : {topic} in {lines} lines.""",
  input_variables=['role', 'topic', 'lines']
)

st.header('Topic Summarizer')
# st.title('Write your query here: ')
user_input = st.text_input('Write your query here')

if st.button('Summarize'):
  # chain1 = template1 | model | parser 
  # result = chain1.invoke({'user_input': user_input})
  # print(type(result))

  # # its a pydanctic object, convert into python dict
  # summary_dict = dict(result)
  
  # chain2 = template2 | model | StrOutputParser()
  # final_result = chain2.invoke({
  #   'role': summary_dict['role'],
  #   'user_question': summary_dict['topic'],
  #   'lines': summary_dict['lines']
  # })

  # used two chains here, because i wanted to see intermediate result of chain 1
  # can use a single chain to get the answer.

  chain = template1 | model | parser | (lambda x: x.model_dump()) | template2 | model | StrOutputParser()
  final_result = chain.invoke(user_input)

  st.write(final_result)
  



key_points = """
    - can use a single chain for all the tasks
    - after extracting, we get a pydantic object that needs to be converted into a dict for next prompt template to understand, as prompt template expects key value pairs.
    - to convert into python dictionary use this method: 
        - model.model_dump()
    - to print the intermediate pydantic object while keeping the same chain, we convert it into a tuple, where 1st element is printing, and second element is converting it into a dictionary:
      - (lambda x : (print(x), x.model_dump())[1])
      - here [1] means, select the 2nd element of the tuple.
    - name of attributes mentioned in pydantic class, should match exactly with input variable names inside the template, otherwise it'll throw an error saying, name need to be same.
"""


