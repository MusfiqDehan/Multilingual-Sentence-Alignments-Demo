"""
This module contains the functions to get PoS tags using Spacy and return a Markdown table
"""

from .alignment_mappers import get_alignment_mapping, select_model

from flair.models import SequenceTagger
from flair.data import Sentence

import spacy
from spacy.cli import download
download("en_core_web_sm")
import en_core_web_sm

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from textblob import TextBlob


def get_spacy_postag_dict(target=""):
    ''' 
    Get spacy pos tags 
    '''
    nlp = en_core_web_sm.load()
    target_tokenized = nlp(target)
    spacy_postag_dict = dict((token.text, token.tag_)
                             for token in target_tokenized)
    return spacy_postag_dict

def get_nltk_postag_dict(target=""):
    ''' 
    Get nltk pos tags 
    '''
    target_tokenized = nltk.tokenize.word_tokenize(target)
    nltk_postag_dict = dict((key, value)
                            for key, value in nltk.pos_tag(target_tokenized))
    return nltk_postag_dict

def get_flair_postag_dict(target=""):
    ''' 
    Get flair pos tags 
    '''
    tagger = SequenceTagger.load("pos")
    target_tokenized = Sentence(target)
    tagger.predict(target_tokenized)
    flair_postag_dict = dict((token.text, token.tag)
                             for token in target_tokenized)
    return flair_postag_dict

def get_textblob_postag_dict(target=""):
    ''' 
    Get textblob pos tags 
    '''
    blob = TextBlob(target)
    textblob_postag_dict = dict(blob.tags)
    return textblob_postag_dict

def get_postag(
        get_postag_dict,
        source="", 
        target="", 
        model_name="musfiqdehan/bn-en-word-aligner"):
    """Get Spacy PoS Tags and return a Markdown table"""

    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_name=model_name
    )
    postag_dict = get_postag_dict(target=target)

    mapped_sent_src = []

    html_table = '''
                    <table>
                        <thead>
                            <th>Bangla</th>
                            <th>English</th>
                            <th>PoS Tags</th>
                        </thead>
                '''

    for i, j in sorted(align_words):
        punc = r"""!()-[]{}।;:'"\,<>./?@#$%^&*_~"""
        if sent_src[i] in punc or sent_tgt[j] in punc:
            mapped_sent_src.append(sent_src[i])

            html_table += f'''
                            <tbody>
                                <tr>
                                    <td> {sent_src[i]} </td>
                                    <td> {sent_tgt[j]} </td>
                                    <td> PUNC </td>
                                </tr>
                            '''
        else:
            mapped_sent_src.append(sent_src[i])

            html_table += f'''
                            <tr>
                                <td> {sent_src[i]} </td>
                                <td> {sent_tgt[j]} </td>
                                <td> {postag_dict[sent_tgt[j]]} </td>
                            </tr>
                            '''

    unks = list(set(sent_src).difference(set(mapped_sent_src)))
    for word in unks:

        html_table += f'''
                        <tr>
                            <td> {word} </td>
                            <td> N/A </td>
                            <td> UNK </td>
                        </tr>                         
                    '''
        
    html_table += '''
                        </tbody>
                    </table>
                '''
    
    pos_accuracy = ((len(sent_src) - len(unks)) / len(sent_src))
    pos_accuracy = f"{pos_accuracy:0.2%}"

    return html_table, pos_accuracy


def select_pos_tagger(src, tgt, model_name, tagger):
    ''' 
    Select the PoS tagger 
    '''

    result = None
    pos_accuracy = None

    model_name = select_model(model_name)

    if tagger == "spaCy":
        result, pos_accuracy = get_postag(
            get_spacy_postag_dict,
            source=src,
            target=tgt,
            model_name=model_name, 
        )
    elif tagger == "NLTK":
        result, pos_accuracy = get_postag(
            get_nltk_postag_dict,
            source=src,
            target=tgt,
            model_name=model_name, 
        )
    elif tagger == "Flair":
        result, pos_accuracy = get_postag(
            get_flair_postag_dict,
            source=src,
            target=tgt,
            model_name=model_name, 
        )
    elif tagger == "TextBlob":
        result, pos_accuracy = get_postag(
            get_textblob_postag_dict,
            source=src,
            target=tgt,
            model_name=model_name, 
        )
    return result, pos_accuracy
