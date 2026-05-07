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
</CONTEXT>

{system_message_content}
"""  # noqa: E501
adamic_input_user_prompt = """
Answer the English question, using the {detected_language} question for context.

Question in English:
{question_en}

Same Question in {detected_language}:
{question_none_en}
"""

adamic_output_system_prompt = """
<GOAL>
Generate a high-quality translation of the given text from English to {detected_language}.
The translation should be accurate, fluent, and maintain the original meaning of the text.
</GOAL>

<CONTEXT>
You will be provided with a question answer example, use that as guide for the translation.
</CONTEXT>

<REQUIREMENTS>
- Output ONLY the translation. No preamble, quotes, explanations, or notes.
- The translation should be in {detected_language} and should not contain any English words or phrases.
</REQUIREMENTS>
"""  # noqa: E501

adamic_output_user_prompt = """
<QUESTION>
Translate the following text from English to {detected_language}:
{question_en}
</QUESTION>

<ANSWER>
{question_none_en}
</ANSWER>

<QUESTION>
Translate the following text from English to {detected_language}:
{answer_en}
</QUESTION>
"""
