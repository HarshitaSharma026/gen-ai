# GenAI projects Repo

This repo holds the GenAI projects that I have developed. 

## Project 1 - Summarizer
This project has been developed to understand the working or PromptTemplate class of Langchain, and different output parsers provided in Langchain. It takes user input, extracts 3 key information from the input namely, "role" the user want AI to play, "topic" to explain and number of "lines" for the answer.
Three different versions are there:
- text_summarizer1: uses JsonOutputParser() to extract the important key findings from the user question and finally uses StrOutputParser() to print the final output. Observations and problems in this version is mentioned at the end of the code.

- text_summarizer2: uses StructuredOutputParser() to structure the schema we are expecting. Observations and problems in this version is mentioned at the end of the code.

- text_summarizer3: uses PydanticOutputParer() to perform data validation. Observations and problems in this version is mentioned at the end of the code.
