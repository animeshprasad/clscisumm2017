#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author : Animesh Prasad
# Copyright : WING - NUS


import re
import os
import unicodedata
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree as ET
import random
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


def preprocess(instring, marker=None):

    def replace(s):
        s = re.sub('[0-9]+', ' T-INT ', s)
        s = re.sub('[0-9.]+', ' T-FLOAT ', s)
        s = re.sub('- ', '', s)
        return s

    marker=r"\([\w+]+\)"
    instring = re.sub(marker, '', instring)
    instring = unicodedata.normalize('NFKD', unicode(instring,'utf-8'))

    return replace(instring)

def parse_file(foldername, file, type='rdata'):
    """
    parses a CLSciSumm training file
    :return: dict of lines as list of tokens
    """
    to_return = {}
    to_return_list = []

    if type == 'rdata':
        base = '/Reference_XML/'
        try:
            f = open(foldername + '/' + file + base + file + '.xml', 'r')
            for lines in f:
                try:
                    tokens = word_tokenize(preprocess(ET.fromstring(lines).text.lower()))
                    sid = ET.fromstring(lines).get('sid')
                    to_return[sid] = tokens
                except:
                    continue
        except:
            pass
        return to_return

    elif type == 'cdata':
        base = '/Citance_XML/'
        try:
            xmls = os.listdir(foldername + '/' + file + base )
            for files in xmls:
                try:
                    f = open(foldername + '/' + file + base + files, 'r')
                    for lines in f:
                        try:
                            tokens = word_tokenize(preprocess(ET.fromstring(lines).text.lower()))
                            sid = ET.fromstring(lines).get('sid')
                            to_return[files[:-4]][str(sid)] = tokens
                        except:
                            continue
                except:
                    continue
        except:
            pass
        return to_return



    elif type == 'annot':
        base = '/annotation/'
        # Arranged in priority as inverse frequency
        facet_types=['Hypothesis_Citation', 'Aim_Citation',
                     'Implication_Citation',  'Results_Citation', 'Method_Citation']
        try:
            f = open(foldername + '/' + file + base + file + '.ann.txt', 'r')
            for lines in f:
                if lines.strip() == '':
                    continue
                ref_lines = []
                ref_text = []
                citance_text = []
                facets = []
                try:
                    components = lines.split('|')
                    if len(components) != 12:
                        # Error caused by annotation delimeter | as part of text
                        print 'Annotation errors in files ' + components[1] + ' at ' + components[0]
                        continue
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[8]):
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower()))
                        sid = ET.fromstring(xml_statements).get('sid')
                        ref_lines.append(sid)
                        ref_text.append(tokens)
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[6]):
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower()))
                        citance_text.append(tokens)
                    fac = components[9].split(':')[1].strip()
                    if fac in facet_types:
                        facets.append(fac)
                    else:
                        #Change
                        for single_fac in facet_types:
                            if single_fac in fac:
                                facets.append(single_fac)
                except:
                    continue
                if ref_lines == [] or ref_text == [] or citance_text == [] or facets == []:
                    print 'Annotation errors in files ' + components[1] + ' at ' + components[0] + ' type 2'
                else:
                    to_return_list.append((ref_lines, ref_text, citance_text, facets))
        except:
            pass
        return to_return_list


    elif type == 'tdata':
        base = '/annotation/'
        # Arranged in priority as inverse frequency
        facet_types=['Hypothesis_Citation', 'Aim_Citation',
                     'Implication_Citation',  'Results_Citation', 'Method_Citation']
        try:
            f = open(foldername + '/' + file + base + file + '.ann.txt', 'r')
            for lines in f:
                if lines.strip() == '':
                    continue
                citance_text = []
                try:
                    components = lines.split('|')
                    if len(components) < 7:
                        # Error caused by annotation delimeter | as part of text
                        print 'Annotation errors in files ' + components[1] + ' at ' + components[0]
                        continue
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[6]):
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower()))
                        citance_text.append(tokens)
                except:
                    continue
                if citance_text == []:
                    print 'Annotation errors in files ' + components[1] + ' at ' + components[0] + ' type 2'
                else:
                    to_return_list.append(citance_text)
        except:
            pass
        return to_return_list

    return -1


def read_training_files(foldername, type='rdata', split=None):
    """
    reads training files in a dictionary of dictionary
    with filename as key for the primary dictionary
    and value being all the lines in that file as dictionary
    with the line number as key.

    Assumes folder for training file of format
    folder/<FILENAME>/Reference_XML/<FILENAME>.xml

    :return: dict of files as dict of lines
    """
    to_return = {}

    folders = os.listdir(foldername)

    #Manual Split for development

    if split == 'train':
        folders = ["N01-1011",  "P98-2143",  "J00-3003",  "P05-1004",  "W04-0213",
            "X96-1048",  "W03-0410",  "E09-2008",  "I05-5011",  "P98-1046",
            "P06-2124",  "C94-2154",  "C04-1089",  "J98-2005",  "N04-1038",
            "C08-1098",  "C02-1025",  "W08-2222",  "E03-1020",  "N06-2049",
                   "P05-1053", "J96-3004", "H89-2014", "D10-1083", "C10-1045",]
    elif split == 'dev':
        folders = [
            "C00-2123",  "C90-2039",  "H05-1115",  "W95-0104",  "P98-1081",]

    if type == 'rdata' or type == 'annot':
        for file in folders:
            to_return[file] = parse_file(foldername, file, type)
        return to_return

    elif type == 'tdata':
        for file in folders:
            to_return[file] = parse_file(foldername, file, type)
        return to_return

    elif type == 'cdata':
        for file in folders:
            to_return[file] = parse_file(foldername, file, type)
        return to_return


def get_data(task='1', split=None):
    """
    :param task:
    :return:
    """
    all_ref_files = read_training_files("../data/Training-Set-2017", 'rdata', split)
    all_tagged = read_training_files("../data/Training-Set-2017", 'annot', split)

    all_ref_files_test=[]
    all_tagged_test = []
    if split is None:
        all_ref_files_test = read_training_files("../data/Test-Set-2017", 'rdata')
        all_tagged_test = read_training_files("../data/Test-Set-2017", 'tdata')
    if split == 'dev':
        all_ref_files_test = read_training_files("../data/Training-Set-2017", 'rdata', split)
        all_tagged_test = read_training_files("../data/Training-Set-2017", 'tdata', split)

    assert(len(all_ref_files) == len(all_tagged))
    assert (len(all_ref_files_test) == len(all_tagged_test))

    firstt = []
    secondt = []
    citt = []
    line_no=[]

    first = []
    second = []
    cit = []
    lab = []


    facet_types = ['Hypothesis_Citation', 'Aim_Citation',
                   'Implication_Citation', 'Results_Citation', 'Method_Citation']
    counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def make_samples():

        def flatten(list_of_lists, use=True):
            if not use:
                return list_of_lists
            else:
                return sum(list_of_lists,[])



        for filenames in all_ref_files_test:
            print 'Parsing for ' + filenames

            for i, citances in enumerate(all_tagged_test[filenames]):
                for c1 in xrange(200):
                    try:
                        a = set(flatten([all_ref_files_test[filenames][str(c1)]])) & set(flatten(citances))
                        if a.issubset(stop):
                            continue
                    except:
                        continue
                    for c2 in xrange(200):
                        if c1 == c2:
                            continue
                        try:
                            # helps by splitting
                            sample_line1 = [all_ref_files_test[filenames][str(c1)]]
                            sample_line2 = [all_ref_files_test[filenames][str(c2)]]
                            firstt.append(flatten(sample_line1))
                            secondt.append(flatten(sample_line2))
                            citt.append(flatten(citances))
                            line_no.append(str(c1)+'_'+str(c2)+'_'+str(i)+'_'+filenames)
                        except:
                            continue


        for filenames in all_ref_files:
            print 'Parsing for ' + filenames

            for ref_id, ref_lines, cit_lines, possible_facets in all_tagged[filenames]:
                positive_samples_ref=[]
                positive_samples_cit=[]
                facets_labels=[]
                negative_samples_ref=[]
                negative_samples_cit=[]

                for facet in possible_facets:
                    positive_samples_ref.append(flatten(ref_lines))
                    positive_samples_cit.append(flatten(cit_lines))
                    facets_labels.append(facet_types.index(facet))

                true_samples = [int(sid) for sid in ref_id]
                if len(true_samples) < 6:
                    counter[len(true_samples)] += 1
                else:
                    counter[5] += 1

                class Counter:
                    # To include samples of more than 1 line provanace
                    def __init__(self, sid_max, done=1):
                        self.MAX_RANGE = 1
                        self.max = sid_max
                        self.done = done
                        self.range = 1

                    def __iter__(self):
                        return self

                    def next(self):  # Python 3: def __next__(self)
                        if self.done == self.max:
                            raise StopIteration
                        else:
                            index_list = []
                            if self.range > self.MAX_RANGE:
                                self.range = 1
                                self.done += 1
                            for index in xrange(self.done, self.done + self.range):
                                index_list.append(index)
                            self.range += 1
                            return index_list

                for c in Counter(250):
                    if set(c) & set(true_samples):
                        continue
                    else:
                        try:
                            #helps by splitting
                            neg_list = [all_ref_files[filenames][str(index)] for index in c]
                        except:
                            continue
                        if split == 'train':
                            if random.randint(0, 100) <= 20:
                                negative_samples_ref.append(flatten(neg_list))
                                negative_samples_cit.append(flatten(cit_lines))
                        else:
                            negative_samples_ref.append(flatten(neg_list))
                            negative_samples_cit.append(flatten(cit_lines))

                for items in negative_samples_ref:
                    first.append(items)
                    second.append(positive_samples_ref[0])
                    cit.append(positive_samples_cit[0])
                    lab.append(0)
                    second.append(items)
                    first.append(positive_samples_ref[0])
                    cit.append(positive_samples_cit[0])
                    lab.append(1)


                # print ref_id, ref_lines, cit_lines
                # print positive_samples_ref, positive_samples_cit
                # print negative_samples_ref, negative_samples_cit

    make_samples()
    print counter

    # labels for task 1


    if task == '1':
        print len(first)
        assert (len(first) == len(second))
        return first, second, cit, lab
    if task == '1t':
        print len(firstt)
        assert (len(firstt) == len(secondt))
        assert (len(firstt) == len(citt))
        assert (len(firstt) == len(line_no))
        return firstt, secondt, citt, line_no

if __name__ == "__main__":
    get_data()

