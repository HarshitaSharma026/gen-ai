# GenAI projects Repo

This repo holds the GenAI projects that I have developed. 

## Project 1 - Summarizer
**Topics Covered: Langchain Output Parsers, Prompt Templates** 
This project has been developed to understand the working or PromptTemplate class of Langchain, and different output parsers provided in Langchain. It takes user input, extracts 3 key information from the input namely, "role" the user want AI to play, "topic" to explain and number of "lines" for the answer.
Three different versions are there:
- text_summarizer1: uses JsonOutputParser() to extract the important key findings from the user question and finally uses StrOutputParser() to print the final output. Observations and problems in this version is mentioned at the end of the code.

- text_summarizer2: uses StructuredOutputParser() to structure the schema we are expecting. Observations and problems in this version is mentioned at the end of the code.

- text_summarizer3: uses PydanticOutputParer() to perform data validation. Observations and problems in this version is mentioned at the end of the code.

## Project 2 - Feedback Analyzer
**Topics Covered: Langchain Chains, parallel and conditional** 
This project takes employee feedback for a company and depending on the sentiment of the feedback suggest three actionable advices to improve the situation. Two different versions of this models has been developed. 
- v1: parallel chains have been implemented, that parallely finds the sentiment and the summary of  the user feedback. This sentiment and summary is being taken to further produce the final suggestions.
- v2: here the suggestions are based on the sentiments. If the sentiment is positive, a different chain will be executed to identify the key strengths mentioned in the feedback and suggest ways to maintain or enhance them. If its negative, another chain is executed to suggest the improvements. 