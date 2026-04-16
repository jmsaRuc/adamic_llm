from google.cloud import translate_v2 as translate
from dotenv import load_dotenv

load_dotenv("benchmark.env")


dic_list_langs = {
    "mgsm": ["en", "de", "ru", "fr", "zh", "es", "ja", "sw", "th", "bn", "te"],
    "xcopa": ["zh", "it", "vi", "tr", "id", "sw", "th", "et", "ta", "ht", "qu"],
    "xnli": [
        "en",
        "de",
        "ru",
        "fr",
        "zh",
        "es",
        "vi",
        "tr",
        "sw",
        "ar",
        "el",
        "th",
        "bg",
        "hi",
        "ur",
    ],
    "paws-x": ["en", "de", "fr", "zh", "es", "ja", "ko"],
    "xlsum": ["en", "fr", "zh", "es", "vi", "tr"],
    "mkqa": ["en", "de", "ru", "fr", "zh_cn", "es", "ja", "vi", "tr", "th"],
    "mmmlu": [
        "ar",
        "bn",
        "de",
        "es",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "pt",
        "sw",
        "yo",
        "zh_cn",
    ],
    "global_mmlu": [
        "de",
        "en",
        "fa",
        "fr",
        "he",
        "ja",
        "ne",
        "pl",
        "si",
        "sn",
        "so",
        "sr",
        "sv",
        "yo",
    ],
    "shareGPT": ["ja", "zh", "es", "fr", "vi", "id", "ko", "ro", "uk", "no"],
    "shareGPT_filter": ["ja", "zh", "es", "fr", "ko"],
}

langs = [
    "zh",
    "zh_cn",
    "it",
    "es",
    "vi",
    "ar",
    "et",
    "tr",
    "el",
    "qu",
    "bg",
    "te",
    "bn",
    "sw",
    "ta",
    "ht",
    "id",
    "de",
    "ur",
    "ja",
    "hi",
    "en",
    "th",
    "fr",
    "ko",
    "ru",
    "pt",
    "he",
    "ne",
    "pl",
    "si",
    "sn",
    "so",
    "sr",
    "sv",
    "yo",
    "fa",
]

lang_codes = {
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "it": "Italian",
    "es": "Spanish",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "et": "Estonian",
    "tr": "Turkish",
    "el": "Greek",
    "qu": "Quechua",
    "bg": "Bulgarian",
    "te": "Telugu",
    "bn": "Bengali",
    "sw": "Swahili",
    "ta": "Tamil",
    "ht": "Haitian Creole",
    "id": "Indonesian",
    "de": "German",
    "ur": "Urdu",
    "ja": "Japanese",
    "hi": "Hindi",
    "en": "English",
    "th": "Thai",
    "fr": "French",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "ms": "Malay",
    "fi": "Finnish",
    "nl": "Dutch",
    "cs": "Czech",
    "uk": "Ukrainian",
    "sv": "Swedish",
    "ro": "Romanian",
    "no": "Norwegian",
    "so": "Somali",
    "sr": "Serbian",
    "sn": "Shona",
    "si": "Sinhala",
    "pl": "Polish",
    "fa": "Persian",
    "ne": "Nepali",
    "he": "Hebrew",
    "yo": "Yoruba",
}


def get_translation_google(text: str, dest: str) -> str:
    """Translates text into the target language.

    Args:
        text: The text to translate.
        dest: The ISO 639-1 code for the target language (e.g., "es", "fr").
        project_id: Your Google Cloud project ID.

    Returns:
        The translated text.
        Returns original text if translation fails.
    """
    try:
        translate_client = (
            translate.Client()
        )  # Assumes GOOGLE_APPLICATION_CREDENTIALS is set

        # The result is a dictionary or list of dictionaries
        result = translate_client.translate(
            text,
            target_language=dest,
            # source_language="en" # Optional: Specify source language
        )

        # print(f"Raw API response: {result}") # See the full structure

        # Extract the translated text
        # For single input, result is a dict:
        if isinstance(result, dict):
            translated_text = result["translatedText"]
            detected_language = result.get(
                "detectedSourceLanguage", "N/A"
            )  # v2 might provide detected language
            #  print(f"Detected source language: {detected_language}")
            return translated_text
        # For multiple inputs (if text were a list), result is a list:
        elif isinstance(result, list) and len(result) > 0:
            translated_text = result[0]["translatedText"]  # Get first translation
            detected_language = result[0].get("detectedSourceLanguage", "N/A")
            #  print(f"Detected source language: {detected_language}")
            return translated_text
        else:
            print("Warning: Unexpected translation result format.")
            return text  # Return original on unexpected result

    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Return original text on error
