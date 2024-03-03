"""
This module contains the helper functions to get the word alignment mapping between two sentences.
"""

import torch
import itertools
import transformers
from transformers import logging

# Set the verbosity to error, so that the warning messages are not printed
logging.set_verbosity_warning()
logging.set_verbosity_error()


def select_model(model_name):
    """
    Select Model
    """
    if model_name == "Google-mBERT (Base-Multilingual)":
        model_name="bert-base-multilingual-cased"
    elif model_name == "Neulab-AwesomeAlign (Bn-En-0.5M)":
        model_name="musfiqdehan/bn-en-word-aligner"
    elif model_name == "BUET-BanglaBERT (Large)":
        model_name="csebuetnlp/banglabert_large"
    elif model_name == "SagorSarker-BanglaBERT (Base)":
        model_name="sagorsarker/bangla-bert-base"
    elif model_name == "SentenceTransformers-LaBSE (Multilingual)":
        model_name="sentence-transformers/LaBSE"

    return model_name


def get_alignment_mapping(source="", target="", model_name=""):
    """
    Get Aligned Words
    """
    model_name = select_model(model_name)

    model = transformers.BertModel.from_pretrained(model_name)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # pre-processing
    sent_src, sent_tgt = source.strip().split(), target.strip().split()

    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [
        tokenizer.tokenize(word) for word in sent_tgt]
    
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [
        tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []

    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]

    sub2word_map_tgt = []

    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8

    threshold = 1e-3

    model.eval()

    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[
            2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[
            2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * \
            (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

    align_words = set()

    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    return sent_src, sent_tgt, align_words



def get_word_mapping(source="", target="", model_name=""):
    """
    Get Word Aligned Mapping Words
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_name=model_name)

    result = []

    for i, j in sorted(align_words):
        result.append(f'bn:({sent_src[i]}) -> en:({sent_tgt[j]})')

    return result



def get_word_index_mapping(source="", target="", model_name=""):
    """
    Get Word Aligned Mapping Index
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_name=model_name)

    result = []

    for i, j in sorted(align_words):
        result.append(f'bn:({i}) -> en:({j})')

    return result



