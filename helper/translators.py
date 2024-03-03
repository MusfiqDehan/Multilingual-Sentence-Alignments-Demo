"""
This file contains the functions to translate the text from one language to another.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator, MyMemoryTranslator, MicrosoftTranslator, YandexTranslator, ChatGptTranslator
from .text_preprocess import decontracting_words, space_punc


# Digit Translation
digit_converter = {
    '০': '0',
    '১': '1',
    '২': '2',
    '৩': '3',
    '৪': '4',
    '৫': '5',
    '৬': '6',
    '৭': '7',
    '৮': '8',
    '৯': '9'
}


def get_translated_digit(sentence):
    """
    Translate the digits from Bengali to English
    """
    translated_sentence = []
    for each_letter in sentence:
        if each_letter in digit_converter.keys():
            translated_sentence.append(digit_converter[each_letter])
            # print(digit_converter[each_letter], end="")
        else:
            translated_sentence.append(each_letter)
            # print(each_letter, end="")

    return "".join(each for each in translated_sentence)

def google_translation(sentence: str, source="auto", target="en") -> str:
    """
    Translate a sentence from one language to another using Google Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = GoogleTranslator()
    translated_sentence = translator.translate(
        sentence, source=source, target=target)
    return translated_sentence


def get_better_translation(src=""):
    src_mod = get_translated_digit(src)
    tgt = google_translation(src_mod)
    tgt = decontracting_words(tgt)
    tgt = tgt.replace('rupees', 'takas').replace('Rs', 'takas')
    return tgt


target_lang_dict = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Aragonese": "an",
    "Armenian": "hy",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Basque": "eu",
    "Bavarian": "bar",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bishnupriya Manipuri": "bpy",
    "Bosnian": "bs",
    "Breton": "br",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chechen": "ce",
    "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh-tw",
    "Chuvash": "cv",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian": "ht",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Ido": "io",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Kirghiz": "ky",
    "Korean": "ko",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Lombard": "lmo",
    "Low Saxon": "nds",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Minangkabau": "min",
    "Nepali": "ne",
    "Newar": "new",
    "Norwegian (Bokmal)": "nb",
    "Norwegian (Nynorsk)": "nn",
    "Occitan": "oc",
    "Persian (Farsi)": "fa",
    "Piedmontese": "pms",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Scots": "sco",
    "Serbian": "sr",
    "Serbo-Croatian": "sh",
    "Sicilian": "scn",
    "Slovak": "sk",
    "Slovenian": "sl",
    "South Azerbaijani": "azb",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Volapük": "vo",
    "Waray-Waray": "war",
    "Welsh": "cy",
    "West Frisian": "fy",
    "Western Punjabi": "pnb",
    "Yoruba": "yo",
    "Thai": "th",
    "Mongolian": "mn"
}

def select_target_lang_code(lang):
    """
    Select the target language code
    """
    return target_lang_dict[lang] if lang in target_lang_dict else "en"
