import gradio as gr
from gradio_rich_textbox import RichTextbox

from helper.text_preprocess import space_punc
from helper.alignment_mappers import select_model, get_alignments_table
from helper.translators import select_target_lang_code, google_translation, get_better_translation


def process_alignments(src, language_name, model_name):
    """
    Bangla PoS Tagger
    """

    tgt = None
    tgt_base = None
    html_table = None
    
    src = space_punc(src)

    tgt_lang_code = select_target_lang_code(language_name)

    tgt = get_better_translation(src, tgt_lang_code)

    tgt = space_punc(tgt)

    tgt_base = google_translation(src, tgt_lang_code)

    model_name = select_model(model_name)

    html_table, alignment_accuracy = get_alignments_table(
        source=src,
        target=tgt,
        model_name=model_name 
    )

    return tgt_base, html_table, alignment_accuracy
    

with gr.Blocks(css="styles.css") as demo:
    gr.HTML("<h1>Multilingual Sentence Alignments</h1>")
    gr.HTML("<p>Align Parallel Sentences using mBERT</p>")
    gr.HTML("<p><a href='https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages' alt='mBERT Repo'>(Support Sentence Alignments Between 104 Languages)</a></p>")

    with gr.Row():
        with gr.Column():
            inputs = [
                gr.Textbox(
                    label="Enter a Sentence (Auto Detect Language)", 
                ),
                gr.Dropdown(
                    choices=
                    [
                        "Afrikaans",
                        "Albanian",
                        "Arabic",
                        "Aragonese",
                        "Armenian",
                        "Asturian",
                        "Azerbaijani",
                        "Bashkir",
                        "Basque",
                        "Bavarian",
                        "Belarusian",
                        "Bengali",
                        "Bishnupriya Manipuri",
                        "Bosnian",
                        "Breton",
                        "Bulgarian",
                        "Burmese",
                        "Catalan",
                        "Cebuano",
                        "Chechen",
                        "Chinese (Simplified)",
                        "Chinese (Traditional)",
                        "Chuvash",
                        "Croatian",
                        "Czech",
                        "Danish",
                        "Dutch",
                        "English",
                        "Estonian",
                        "Finnish",
                        "French",
                        "Galician",
                        "Georgian",
                        "German",
                        "Greek",
                        "Gujarati",
                        "Haitian",
                        "Hebrew",
                        "Hindi",
                        "Hungarian",
                        "Icelandic",
                        "Ido",
                        "Indonesian",
                        "Irish",
                        "Italian",
                        "Japanese",
                        "Javanese",
                        "Kannada",
                        "Kazakh",
                        "Kirghiz",
                        "Korean",
                        "Latin",
                        "Latvian",
                        "Lithuanian",
                        "Lombard",
                        "Low Saxon",
                        "Luxembourgish",
                        "Macedonian",
                        "Malagasy",
                        "Malay",
                        "Malayalam",
                        "Marathi",
                        "Minangkabau",
                        "Nepali",
                        "Newar",
                        "Norwegian (Bokmal)",
                        "Norwegian (Nynorsk)",
                        "Occitan",
                        "Persian (Farsi)",
                        "Piedmontese",
                        "Polish",
                        "Portuguese",
                        "Punjabi",
                        "Romanian",
                        "Russian",
                        "Scots",
                        "Serbian",
                        "Serbo-Croatian",
                        "Sicilian",
                        "Slovak",
                        "Slovenian",
                        "South Azerbaijani",
                        "Spanish",
                        "Sundanese",
                        "Swahili",
                        "Swedish",
                        "Tagalog",
                        "Tajik",
                        "Tamil",
                        "Tatar",
                        "Telugu",
                        "Turkish",
                        "Ukrainian",
                        "Urdu",
                        "Uzbek",
                        "Vietnamese",
                        "Volapük",
                        "Waray-Waray",
                        "Welsh",
                        "West Frisian",
                        "Western Punjabi",
                        "Yoruba",
                        "Thai",
                        "Mongolian"
                    ], 
                    label="Select Target Language"
                ),
                gr.Dropdown(
                    choices=[
                        "Google-mBERT (Base-Multilingual)", "SentenceTransformers-LaBSE (Multilingual)"
                    ], 
                    label="Select a Model"
                )
            ]

            btn = gr.Button(value="Submit", elem_classes="mybtn")
            gr.ClearButton(inputs)

        with gr.Column():
            outputs = [
                gr.Textbox(label="English Translation"), 
                RichTextbox(label="Alignments Mapping (Source to Target)"),
                gr.Textbox(label="Alignment Accuracy")
            ]

    btn.click(process_alignments, inputs, outputs)

    gr.Examples([
        [
            "বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র।", 
            "English", 
            "SentenceTransformers-LaBSE (Multilingual)", 
        ],
        [
            "বাংলাদেশের সংবিধানিক নাম কি?", 
            "English", 
            "Google-mBERT (Base-Multilingual)",
        ],
        [
            "বাংলাদেশের সাংবিধানিক নাম গণপ্রজাতন্ত্রী বাংলাদেশ।", 
            "Hindi", 
            "Google-mBERT (Base-Multilingual)",
        ],
        [
            "বাংলাদেশের সংবিধানিক নাম কি?", 
            "Punjabi", 
            "Google-mBERT (Base-Multilingual)",
        ],

    ], inputs)



# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()