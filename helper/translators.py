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


def get_better_translation(translator_func, src=""):
    src_mod = get_translated_digit(src)
    tgt = translator_func(src_mod)
    tgt = decontracting_words(tgt)
    tgt = tgt.replace('rupees', 'takas').replace('Rs', 'takas')
    return tgt

# def select_translator(src, translator):
#     """
#     Select the translator
#     """
#     tgt = None
#     tgt_base = None

#     if translator == "Google":
#         tgt = get_better_translation(google_translation, src)
#         tgt = space_punc(tgt)
#         tgt_base = google_translation(src)
#     elif translator == "BanglaNMT":
#         tgt = get_better_translation(banglanmt_translation, src)
#         tgt = space_punc(tgt)
#         tgt_base = banglanmt_translation(src)
#     elif translator == "MyMemory":
#         tgt = get_better_translation(mymemory_translation, src)
#         tgt = space_punc(tgt)
#         tgt_base = mymemory_translation(src)
    
#     return tgt_base, tgt

# Afrikaans
# Albanian
# Arabic
# Aragonese
# Armenian
# Asturian
# Azerbaijani
# Bashkir
# Basque
# Bavarian
# Belarusian
# Bengali
# Bishnupriya Manipuri
# Bosnian
# Breton
# Bulgarian
# Burmese
# Catalan
# Cebuano
# Chechen
# Chinese (Simplified)
# Chinese (Traditional)
# Chuvash
# Croatian
# Czech
# Danish
# Dutch
# English
# Estonian
# Finnish
# French
# Galician
# Georgian
# German
# Greek
# Gujarati
# Haitian
# Hebrew
# Hindi
# Hungarian
# Icelandic
# Ido
# Indonesian
# Irish
# Italian
# Japanese
# Javanese
# Kannada
# Kazakh
# Kirghiz
# Korean
# Latin
# Latvian
# Lithuanian
# Lombard
# Low Saxon
# Luxembourgish
# Macedonian
# Malagasy
# Malay
# Malayalam
# Marathi
# Minangkabau
# Nepali
# Newar
# Norwegian (Bokmal)
# Norwegian (Nynorsk)
# Occitan
# Persian (Farsi)
# Piedmontese
# Polish
# Portuguese
# Punjabi
# Romanian
# Russian
# Scots
# Serbian
# Serbo-Croatian
# Sicilian
# Slovak
# Slovenian
# South Azerbaijani
# Spanish
# Sundanese
# Swahili
# Swedish
# Tagalog
# Tajik
# Tamil
# Tatar
# Telugu
# Turkish
# Ukrainian
# Urdu
# Uzbek
# Vietnamese
# Volapük
# Waray-Waray
# Welsh
# West Frisian
# Western Punjabi
# Yoruba
# Thai
# Mongolian

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
