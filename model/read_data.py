#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author : Animesh Prasad
# Copyright : WING - NUS


import re
import os
import unicodedata
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree as ET


def preprocess(instring, marker=None):

    def replace(s):
        s = re.sub('[0-9]+', ' ', s)
        s = re.sub('[0-9.]+', ' ', s)
        s = re.sub('-  ', '', s)
        s = re.sub('- ', '', s)
        s = re.sub('-', '', s)
        return s
    if marker is None:
        marker = r"\(.*?\)"
    else:
        pass
        #print marker
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

    if type == 'cdata':
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
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[6]):
                        #print components[4]
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower(), marker=components[4]))
                        sid = ET.fromstring(xml_statements).get('sid')
                        ref_lines.append(sid)
                        ref_text.append(tokens)
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[8]):
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower(), marker=components[4]))
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


    elif type == 'test1b':
        # Arranged in priority as inverse frequency
        facet_types=['Hypothesis_Citation', 'Aim_Citation',
                     'Implication_Citation',  'Results_Citation', 'Method_Citation']
        try:
            f = open(foldername + '/' + file, 'r')
            print 'here'
            for lines in f:
                ref_lines = []
                ref_text = []
                citance_text = []

                try:
                    components = lines.split('|')
                    if len(components) < 10:
                        # Error caused by annotation delimeter | as part of text
                        print 'Annotation errors in files ' + components[1] + ' at ' + components[0]
                        continue
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[6]):
                        print components[4]
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower(), marker=components[4]))
                        sid = ET.fromstring(xml_statements).get('sid')
                        ref_lines.append(sid)
                        ref_text.append(tokens)
                        others = components[0] + ' | ' + components[1]
                    for xml_statements in re.findall('<S.*?>.*?</S>', components[8]):
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower(), marker=components[4]))
                        citance_text.append(tokens)

                except:
                    continue
                if ref_lines == [] or ref_text == [] or citance_text == []:
                    print 'Annotation errors in files ' + components[1] + ' at ' + components[0] + ' type 2'
                else:
                    to_return_list.append((ref_lines, ref_text, citance_text,others ))
        except:
            pass
        return to_return_list



    elif type == 'tdata':
        base = '/annotation/'
        # Arranged in priority as inverse frequency
        facet_types = ['Hypothesis_Citation', 'Aim_Citation',
                       'Implication_Citation', 'Results_Citation', 'Method_Citation']
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
                        tokens = word_tokenize(preprocess(ET.fromstring(xml_statements).text.lower(), marker=components[4]))
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


def read_training_files(foldername, type='rdata'):
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
    if type == 'rdata' or type == 'annot' or type=='tdata':
        for file in folders:
            to_return[file] = parse_file(foldername, file, type)
        return to_return

    elif type == 'cdata':
        for file in folders:
            to_return[file] = parse_file(foldername, file, type)
        return to_return



def get_data(task='1'):
    """
    :param task:
    :return:
    """
    all_tagged ={}
    r_ref=[]
    r_cit=[]
    ot=[]
    if task == 'test1b':
        folders = os.listdir("../data/Test-2017/Task1")
        for file in folders:
            all_tagged[file] = parse_file("../data/Test-2017/Task1", file, 'test1b')
            print file
            for ref_id, ref_lines, cit_lines, others in all_tagged[file]:
                #print ref_id, ref_lines, cit_lines
                r_ref.append(sum(ref_lines,[]))
                r_cit.append(sum(cit_lines,[]))
                ot.append(others)
        #print r_ref, r_cit
        return  r_ref, r_cit, ot







    all_ref_files = read_training_files("../data/Training-Set-2017", 'rdata')
    all_tagged = read_training_files("../data/Training-Set-2017", 'annot')

    # Might be needed
    # all_citances = read_training_files("..data/Training-Set-2017", 'cdata')

    positive_samples_ref = []
    positive_samples_cit = []

    negative_samples_ref = []
    negative_samples_cit = []

    facets_labels = []
    facet_types = ['Hypothesis_Citation', 'Aim_Citation',
                   'Implication_Citation', 'Results_Citation', 'Method_Citation']
    counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def make_samples():

        def flatten(list_of_lists, use=True):
            if not use:
                return list_of_lists
            else:
                return sum(list_of_lists,[])

        for filenames in all_ref_files:
            print 'Parsing for ' + filenames

            for ref_id, ref_lines, cit_lines, possible_facets in all_tagged[filenames]:
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

                for c in Counter(200):
                    if set(c) & set(true_samples):
                        continue
                    else:
                        try:
                            #helps by splitting
                            neg_list = [all_ref_files[filenames][str(index)] for index in c]
                        except:
                            continue
                        negative_samples_ref.append(flatten(neg_list))
                        negative_samples_cit.append(flatten(cit_lines))

                # print ref_id, ref_lines, cit_lines
                # print positive_samples_ref, positive_samples_cit
                # print negative_samples_ref, negative_samples_cit

    make_samples()
    print counter

    # labels for task 1
    plabels = [1]*len(positive_samples_ref)
    nlabels = [0]*len(negative_samples_ref)

    nfacets_labels = [5]*len(negative_samples_ref)

    all_samples_ref = positive_samples_ref + negative_samples_ref
    all_samples_cit = positive_samples_cit + negative_samples_cit
    all_labels = plabels + nlabels
    all_facet_lables = facets_labels + nfacets_labels

    if task == '1a':
        assert (len(negative_samples_ref) == len(negative_samples_cit))
        assert (len(positive_samples_ref) == len(positive_samples_cit))
        assert (len(all_samples_ref) == len(all_labels))
        return all_samples_ref, all_samples_cit, all_labels
    elif task == '1b':
        assert (len(facets_labels) == len(positive_samples_ref))
        assert (len(facets_labels) == len(positive_samples_cit))
        return positive_samples_ref, positive_samples_cit, facets_labels
    elif task == '1':
        print len(all_samples_ref)
        assert (len(all_samples_ref) == len(all_facet_lables))
        return all_samples_ref, all_samples_cit, all_labels, all_facet_lables


if __name__ == "__main__":
    get_data('test1b')



    exit()

    folders = os.listdir('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-Set-2017')
    output = open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-2017/Task1/out', 'r')
    #output_out = open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-2017/final', 'w')
    annot={}
    rlines={}

    for file in folders:
        try:
            temp1 = open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-Set-2017' + '/' + file + '/annotation/' + file + '.ann.txt', 'r')
            annot[file]= temp1.readlines()
            rlines[file]=[]
            tempff = open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-Set-2017' + '/' + file + '/Reference_XML/' + file + '.xml', 'r')
            for lines in tempff:
                try:
                    if ET.fromstring(lines).get('sid'):
                        rlines[file].append(lines.strip())
                except:
                    continue
        except:
            pass

    output_final={}
    for items in output:
        a=items.strip().split()
        if a[0][:8] not in output_final.keys():
            output_final[a[0][:8]] = [[int(k[8:]) for k in a]]
        else:
            output_final[a[0][:8]].append([int(k[8:]) for k in a])


    for file in annot.keys():
        f= open('/Users/animeshprasad/PycharmProjects/scisumm/data/Test-2017/Task1/'+file+'.ann.txt','w')
        for a,b in zip(annot[file],output_final[file]):
            #print a
            #print b
            #raw_input()
            #print a,b
            if len (annot[file])!=len(output_final[file]):
                print 'Exception'

            a=a.strip()
            s=''
            for lin_no in b:
                lin_no = int(lin_no)

                try:
                     for ln in xrange(lin_no-10, lin_no+10):
                        if int(ET.fromstring(rlines[file][ln]).get('sid')) == lin_no:
                            s += rlines[file][ln]
                            break
                except:
                    print 'why'
                    raw_input()
            if a[-1] !='|':
                a=a+'|'
            f.write( a  +  ' Reference Offset: ' + str(b) + ' | ' +
                     ' Reference Text: ' + s + ' | ' + ' Discourse Facet: ' + ' \n')








