from google.cloud import translate_v3 as translate
from google.api_core.retry_async import AsyncRetry
import asyncio
from dotenv import load_dotenv
from datasets import Dataset
import os

from adamic_llm.services.adamic_prompting import state

from adamic_llm.services.adamic_prompting import state

load_dotenv("benchmark.env")

translate_client = translate.TranslationServiceAsyncClient()
retry = AsyncRetry(initial=0.5, maximum=30, multiplier=2, timeout=240)


async def get_translation_google(
    text: dict | Dataset, dest: str, project_id: str
) -> dict | Dataset:
    """Translates text into the target language.

    Args:
        text: The text to translate.
        dest: The ISO 639-1 code for the target language (e.g., "es", "fr").
        project_id: Your Google Cloud project ID.

    Returns:
        The translated text.
        Returns original text if translation fails.
    """
    contents_q = text["Question"]
    contents_a = text["A"]
    contents_b = text["B"]
    contents_c = text["C"]
    contents_d = text["D"]
    try:
        parent = f"projects/{project_id}/locations/global"
        result = await translate_client.translate_text(
            parent=parent,
            contents=[contents_q, contents_a, contents_b, contents_c, contents_d],
            mime_type="text/plain",
            source_language_code=dest,
            target_language_code="en",
            retry=retry,
        )

        # print(f"Raw API response: {result}") # See the full structure

        if (
            result
            and result.translations
            and len(result.translations) > 0
            and len(result.translations) == 5
        ):
            for i, translation in enumerate(result.translations):
                if i == 0:
                    text["Question"] = translation.translated_text
                elif i == 1:
                    text["A"] = translation.translated_text
                elif i == 2:
                    text["B"] = translation.translated_text
                elif i == 3:
                    text["C"] = translation.translated_text
                elif i == 4:
                    text["D"] = translation.translated_text
            return text
        else:
            print("Warning: Unexpected translation result format.")
            return None  # Return original on unexpected result

    except Exception as e:
        print(f"Error during translation: {e}")
        return None  # Return original text on error


async def get_question_translation_google_for_mmmlu(
    question: Dataset, dest_lang_code: str, num_samples: int, project_id: str
) -> Dataset:
    num_samples = min(num_samples, len(question))
    list_idx = list(range(num_samples))
    task = [
        asyncio.create_task(
            get_translation_google(question[i], dest_lang_code, project_id)
        )
        for i in list_idx
    ]
    translated_questions = await asyncio.gather(*task)
    return translated_questions
