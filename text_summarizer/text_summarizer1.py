# A summarizer that take input and summarizes it in the mentioned lines.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


parser = JsonOutputParser()
template1 = PromptTemplate(
  template="""Take the user input: {user_input} and extract the role you have to play, the topic to explain, and the number of lines.\n {format_instruction}""",
  input_variables=['user_input'],
  partial_variables={'format_instruction': parser.get_format_instructions()}
)

template2 = PromptTemplate(
  template="""You are a helpful {role}. Answer the user question : {user_question} in {lines} lines.""",
  input_variables=['role', 'user_question', 'lines']
)


st.header('Topic Summarizer')
# st.title('Write your query here: ')
user_input = st.text_input('Write your query here')

if st.button('Summarize'):
  output1 = template1 | model | parser
  inter_result = output1.invoke({'user_input': user_input})
  print(f"\n\n {inter_result}")

  output2 = template2 | model | StrOutputParser()
  result = output2.invoke({
    'role': inter_result['role'],
    'user_question': inter_result['topic'],
    'lines': inter_result['lines']
  })

  st.write(result)



explanation = """
Important points:
- LLM extracts role, topic and lines along with explanation. (for first prompt)
- if the question is not given in this format "you are a {role}, explain {topic} in {lines} lines. -> it'll take this form:  
{'role': 'Explainer', 'topic': 'Quantum mechanics', 'lines': 1}
    - user JsonOutputParser() to parse json output
"""

problems = """
- not able to provide structure of schema
- let's say i want default value for lines as 4
- and default value for 'role' => 'expert'
- Use StructuredOutputParser for this.
"""
