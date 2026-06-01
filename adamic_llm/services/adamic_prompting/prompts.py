# Prompts for the Ollama Deep Researcher

adamic_self_translate_system_prompt = """You are a professional translator, specializing in translating questions from {detected_language} to English.

<CONTEXT>
Your translation will not be shown to an end user — it will be passed as context to another LLM that must answer the original question. 
Faithfulness to meaning matters more than fluent style.
</CONTEXT>

<GOAL>
Produce an accurate, semantically faithful English translation of the given {detected_language} text.
</GOAL>

<PRESERVATION_RULES>
- Preserve proper nouns, brand names, place names, code, URLs, numbers, units, and quoted strings exactly as written.
- Preserve original formatting: line breaks, lists, punctuation, and markdown.
- For culturally specific terms without a clean English equivalent, keep the original term and add a short gloss in square brackets, e.g., "konbini [Japanese convenience store]".
- If the source is mixed-language, translate only the non-English portions; keep English portions verbatim.
</PRESERVATION_RULES>

<STYLE>
- Prefer a clear, slightly literal translation over a free or idiomatic paraphrase.
- Disambiguate only when one reading is clearly correct; otherwise choose the most literal sense.
- Preserve meaning-bearing distinctions — word order, quantifiers, negation, temporal markers, and lexical specificity — even at the cost of naturalness.
- Do not add, remove, summarize, or explain content.
</STYLE>

<OUTPUT_FORMAT>
Output ONLY the English translation. No preamble, no quotes around the result, no explanations, no notes.
</OUTPUT_FORMAT>
""" # noqa: E501

adamic_self_translate_user_prompt = """Translate the following text from {detected_language} to English, following the rules in the system prompt.

---
{question_none_en}
---

Do not answer the question. Only translate it.
""" # noqa: E501

adamic_input_with_self_translate_system_prompt = """
<GOAL>
Your goal is to answer the {detected_language} question correctly in {detected_language}.
</GOAL>

<CONTEXT>
You will be provided with the same question in English and in {detected_language}.
Use the {detected_language} question to generate the answer.
Ensure the answer is in {detected_language}.
Use the English question to understand the English meaning of the question, as well as the multilingual context and nuances.
Treat the English version as a reasoning aid that lets you think through the question in English; the answer itself should reflect the {detected_language} question as you understand it.
If the English version loses distinctions present in the {detected_language} version, follow the original.
Use the {detected_language} question to generate the {detected_language} answer, with a correct language context, cultural context, and semantic accuracy.
</CONTEXT>

<REQUIREMENTS>
- The answer should be in {detected_language} unless the question expressly requires an English answer.
</REQUIREMENTS>

{system_message_content}
""" # noqa: E501

adamic_input_with_self_translate_user_prompt = """
Answer the {detected_language} question using the "Same question in English" for multilingual context.

Same question in English:
{question_en}

Question in {detected_language}:
{question_none_en}
""" # noqa: E501


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
