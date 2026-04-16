# Prompts for the Ollama Deep Researcher

adamic_input_system_prompt = """
<GOAL>
Your goal is to answer the English question correctly in English.
</GOAL>

<CONTEXT>
You will be provided with the same question in English and in {detected_language}.
Use the English question to generate the answer.
Ensure the answer is in English.
Use the {detected_language} question to generate the English answer with a correct language context and cultural context.
For native terms and concepts that are difficult to translate, use the {detected_language} term in the English answer, dont write that this tearm is in {detected_language}, just use the term as it is.
</CONTEXT>

{system_message_content}
"""  # noqa: E501
adamic_input_user_prompt = """
Answer the English question, using the {detected_language} question for context, dont write any {detected_language} in the answer, only write English:

Question in English:
{question_en}

Same Question in {detected_language}:
{question_none_en}
"""  # noqa: E501

adamic_output_system_prompt = """
<GOAL>
Generate a high-quality translation of the given text from English to {detected_language}.
The translation should be accurate, fluent, and maintain the original meaning of the text.
Use the example provided as a guide for the translation.
</GOAL>

<RQUREMENTS>
- do not add any extra text or explanations
- do not add any extra line breaks
- do not add any extra spaces
- do not add any extra characters
- do not add any extra punctuation
- do not add any extra formatting
- do not add any extra symbols
- do not add any extra tags
</RQUREMENTS>
"""  # noqa: E501

adamic_output_user_prompt = """""
Use the following translation example as a guide on how to translate English to {detected_language} with correct language understanding:

<USER>
Translate the following text from English to {detected_language}:
{question_en}
</USER>

<ASSISTANT>
{question_none_en}
</ASSISTANT>

<USER>
Translate the following text from English to {detected_language}:
{answer_en}
</USER>
"""  # noqa: E501
