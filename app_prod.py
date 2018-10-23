#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Anna Klezovich

from flask import Flask
from flask import render_template, redirect, url_for, request

import pandas as pd
import numpy as np
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log
import string
import re
import os
import json

morph = pymorphy2.MorphAnalyzer()

app = Flask(__name__)


def get_names():
    with open("file_names.txt", "r") as file:
        k = file.readlines()
    return k


texts = get_names()


def score_BM25(qf, dl, N, n, avgdl, k1=2.0, b=0.75) -> float:
    score = log((N - n + 0.5)/(n + 0.5)) * (k1 + 1) * qf / (qf + k1 * (1 - b + b * (dl / avgdl)))
    return score


inv_ind = json.load(open('invind.json'))


@app.route('/',  methods=['GET'])
def index():
    if request.args:
        query = request.args.get('query')
        search_method = request.args.get('search_method')
        search_result = search(query, search_method)
        return render_template('result.html', df=search_result.to_html(), text=search_result.text, query=query)
    return render_template('index.html')  # test that texts loaded


# MAIN FUNCTION
def search(query, search_method, inv_ind):
    if search_method == 'inverted_index':
        search_result = get_search_result(query, inv_ind, texts)
    else:
        raise TypeError('Unsupported search method')
    return search_result[:5]  # return top-5


def compute_sim(query, inv_ind, document, avgdl, N) -> float:
    if query in inv_ind.keys():
        n = len(inv_ind[query])
    else:
        n = 0
    qf = document.count(query)
    dl = len(document)
    scores = score_BM25(qf, dl, N, n, avgdl)
    return scores


def get_search_result(query, inv_ind, doc_text) -> list:
    avgdl = 759
    N = 9486
    lemmas_query = preprocessing(query, del_stopwords=False)
    result = []
    i = 0
    for document in doc_text:
        similar = 0
        for el in lemmas_query:
            similar += compute_sim(el, inv_ind, document, avgdl, N)
        result.append((doc_text[i], similar))
        i += 1
    res = pd.DataFrame(result, columns=['text', 'similarity'])
    return res.sort_values('similarity', ascending=False)


def preprocessing(input_text, del_stopwords=True, del_digit=True):
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(string.punctuation+'»«–…') for x in word_tokenize(input_text)]
    lemmas = [morph.parse(x)[0].normal_form for x in words if x]
    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords or lemma == "n":
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr


# удаляем из текста символы переноса строки
# убираем пробелы перед сокращенными названиями населенных пуктов
def no_spaces(text):
    processed_text = text.replace('\n', ' ').replace('\n\n', ' ').replace('ул. ', 'ул.').replace('г. ', 'г.').replace(
        'гор. ', 'гор.').replace('с. ', 'с.')
    return processed_text


# убираем пробел после инициалов перед фамилией
def clear_abbrs(processed_text):
    initials = re.compile(r'[А-Я]{1}\.[А-Я]{1}\. [А-Я][а-яё]+')
    counter = len(initials.findall(processed_text))

    for s in range(counter):
        get_abbrs = initials.search(processed_text)
        i = get_abbrs.span()[0] + 4
        processed_text = processed_text[:i] + processed_text[i + 1:]
    return processed_text


# делим текст на предложения при помощи регулярного выражения
def split_text(processed_text):
    text_splitted = re.split(r'(\. +[А-Я]{1} *[а-яё]+)', processed_text)
    last_word = re.compile(r'[А-Я]{1} *[а-яё]+')
    normal_sentences = [text_splitted[0] + '.']

    for i in range(1, len(text_splitted), 2):
        if i + 1 <= len(text_splitted) - 1:
            beginning = last_word.findall(text_splitted[i])[0]
            normal_sentences.append(beginning + text_splitted[i + 1] + '.')
        elif i == len(text_splitted) - 1:
            beginning = last_word.findall(text_splitted[i])[0]
            normal_sentences.append(beginning)
    return normal_sentences


def get_sentences(text):
    text = no_spaces(text)
    text = clear_abbrs(text)
    sentences = split_text(text)
    return sentences


# делим текст на куски по n предложений
# (функция принимает на вход список из предложений-строк, полученный на предыдущем шаге)
def split_paragraph(list_of_sentences, n):
    l = len(list_of_sentences)

    n_chunks = []
    chunk = ''

    for i in range(0, l, n):
        for j in range(n):
            if i + j < l:
                chunk += list_of_sentences[i + j] + ' '
            else:
                continue
        n_chunks.append(chunk)
        chunk = ''
    return n_chunks


def splitter(text, n):
    normal_sentences = get_sentences(text)
    split_sentences = split_paragraph(normal_sentences, n)
    return split_sentences


if __name__ == '__main__':
    app.run(debug=False)
