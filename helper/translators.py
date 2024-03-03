"""
This file contains the functions to translate the text from one language to another.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator, MyMemoryTranslator, MicrosoftTranslator, YandexTranslator, ChatGptTranslator
from .text_preprocess import decontracting_words, space_punc
from dotenv import load_dotenv
import os


# Load the environment variables from the .env file
load_dotenv()

# Translators API Keys
MICROSOFT_API_KEY = os.getenv("MICROSOFT_TRANSLATOR_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

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

# Bangla to English Translation (BUET BanglaNMT)
translation_model_bn_en = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_bn_en")
translation_tokenizer_bn_en = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_bn_en")

def banglanmt_translation(input_text):
    """
    Translate a sentence from Bengali to English using BUET BanglaNMT
    """
    inputs = translation_tokenizer_bn_en(input_text, return_tensors="pt")
    outputs = translation_model_bn_en.generate(**inputs)
    translated_text = translation_tokenizer_bn_en.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def google_translation(sentence: str, source="bn", target="en") -> str:
    """
    Translate a sentence from one language to another using Google Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = GoogleTranslator()
    translated_sentence = translator.translate(
        sentence, source=source, target=target)
    return translated_sentence

def microsoft_translation(sentence: str, source="bn", target="en") -> str:
    """
    Translate a sentence from one language to another using Microsoft Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = MicrosoftTranslator(api_key=MICROSOFT_API_KEY, target='en')
    translated_sentence = translator.translate(sentence)
    return translated_sentence

def chatgpt_translation(sentence: str, source="bn", target="en") -> str:
    """
    Translate a sentence from one language to another using ChatGPT Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = ChatGptTranslator(api_key=OPENAI_API_KEY, target=target)
    translated_sentence = translator.translate(sentence)
    return translated_sentence

def yandex_translation(sentence: str, source="bn", target="en") -> str:
    """
    Translate a sentence from one language to another using Yandex Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = YandexTranslator(api_key=YANDEX_API_KEY)
    translated_sentence = translator.translate(
        sentence, source=source, target=target)
    return translated_sentence

def mymemory_translation(sentence: str, source="bn-IN", target="en-US") -> str:
    """
    Translate a sentence from one language to another using MyMemory Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`
    """
    translator = MyMemoryTranslator(source=source, target=target)
    translated_sentence = translator.translate(sentence)
    return translated_sentence

def get_better_translation(translator_func, src=""):
    src_mod = get_translated_digit(src)
    tgt = translator_func(src_mod)
    tgt = decontracting_words(tgt)
    tgt = tgt.replace('rupees', 'takas').replace('Rs', 'takas')
    return tgt

def select_translator(src, translator):
    """
    Select the translator
    """
    tgt = None
    tgt_base = None

    if translator == "Google":
        tgt = get_better_translation(google_translation, src)
        tgt = space_punc(tgt)
        tgt_base = google_translation(src)
    elif translator == "BanglaNMT":
        tgt = get_better_translation(banglanmt_translation, src)
        tgt = space_punc(tgt)
        tgt_base = banglanmt_translation(src)
    elif translator == "MyMemory":
        tgt = get_better_translation(mymemory_translation, src)
        tgt = space_punc(tgt)
        tgt_base = mymemory_translation(src)
    
    return tgt_base, tgt
