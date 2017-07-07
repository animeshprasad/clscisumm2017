import pickle
import os
import math
import operator
from read_data import parse_file, read_training_files


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

class tfidf:
  def __init__(self):
    self.weighted = False
    self.documents = []
    self.corpus_dict = {}
    self.idf = {}

  def addDocument(self, doc_name, list_of_words):
    # building a dictionary
    doc_dict = {}
    for w in list_of_words:
      doc_dict[w] = doc_dict.get(w, 0.) + 1.0
      self.corpus_dict[w] = self.corpus_dict.get(w, 0.0) + 1.0

    # normalizing the dictionary
    length = float(len(list_of_words))
    for k in doc_dict:
      doc_dict[k] = doc_dict[k] / length

    # add the normalized document to the corpus
    self.documents.append([doc_name, doc_dict])

  def get_idf(self):
    """Print dictionary"""
    total_doc=len(self.documents)
    for items in self.corpus_dict:
        count =0
        for id, docs in self.documents:
            if docs.get(items,0) != 0:
                count += 1
        self.idf[items] = math.log(total_doc+1/(count + 1.0))

  def print_dictionary(self):
    """Print dictionary"""
    for key,items in self.documents:
        print key,items
    for items in self.corpus_dict:
        print items

  def get_tfidf(self,str):
    """"Get TFIDF"""


  def save(self,path):
      save_obj(self.documents, path + 'documents')
      save_obj(self.idf, path + 'idf')


  def similarities(self, list_of_words, doc_subset):
    """Returns a list of all the [docname, similarity_score] pairs relative to a list of words."""

    # building the query dictionary
    query_dict = {}
    for w in list_of_words:
      query_dict[w] = query_dict.get(w, 0.0) + 1.0

    # normalizing the query
    length = float(len(list_of_words))
    for k in query_dict:
      query_dict[k] = query_dict[k] / length

    # computing the list of similarities
    #sims = []
    sims = {}
    for doc in self.documents:
      if doc[0][:8] != doc_subset:
          continue
      score = 0.0
      doc_dict = doc[1]
      for k in query_dict:
        if k in doc_dict:
          score += (query_dict[k] / self.corpus_dict[k]) * (doc_dict[k] / self.corpus_dict[k])
      #sims.append([doc[0], score])
      sims[doc[0]] = score

      jscore = 0.0
      score = 0.0
      for k in query_dict:
        if k in doc_dict:
          score += (query_dict[k] / self.corpus_dict[k]) * (doc_dict[k] / self.corpus_dict[k])
      for k in query_dict:
          try:
            jscore += (query_dict[k] / self.corpus_dict[k])
          except:
            jscore += (query_dict[k] / self.corpus_dict['verbmobil'])   # A very rare word
      for k in doc_dict:
          jscore += (doc_dict[k] / self.corpus_dict[k])
      sims[doc[0]] = score / jscore


    return sims


def flatten(list_of_lists, use=True):
    if not use:
        return list_of_lists
    else:
        return sum(list_of_lists, [])


def parsexml(foldername = "../data/Training-Set-2017"):
    '''
    Read all .xml files to generate tf-idfs
    :param textfolder:
    :return:
    '''
    folders = os.listdir(foldername)
    document = {}
    ref_documents = {}

    for file in folders:
        if '-' in file:
            all_lines = parse_file(foldername, file, 'rdata')
            l=[]
            for sid in all_lines:
                l.extend(all_lines[sid])
            document[file] = l
            all_lines = parse_file(foldername, file, 'cdata')
            for fid in all_lines:
                for sid in fid:
                    l.extend(all_lines[sid][fid])
            document[file] = l

    return document

def parsexml_lines(foldername = "../data/Training-Set-2017"):
    '''
    Read all .xml files to generate tf-idfs
    Uses each line and combination of lines as documents
    :param textfolder:
    :return:
    '''
    folders = os.listdir(foldername)
    document = {}
    all_ref_files = {}


    for file in folders:
        if '-' in file:
            all_ref_files[file] = parse_file(foldername, file, 'rdata')

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
                try:
                    # helps by splitting
                    doc = [all_ref_files[file][str(index)] for index in c]
                except:
                    continue
                document[file + ''.join(str(index) for index in c)] = flatten(doc)

    return document


if __name__ == '__main__':
    table = tfidf()
    #docs = parsexml("../data/Training-Set-2017")


    docs = parsexml_lines("../data/Training-Set-2017")
    for name in docs:
        table.addDocument(name, docs[name])
    table.get_idf()

    docs = read_training_files("../data/Training-Set-2017", 'annot')
    for filenames in docs:
        for ref_id, ref_lines, cit_lines, facet in docs[filenames]:
            for lines, s in zip(cit_lines, ref_id):
                x = table.similarities(lines, filenames)
                if x.get(filenames + str(s)) == 0:
                    print s
                    print 'Hi'
                true_sample = x.get(filenames + str(s))
                #x = table.similarities(flatten(cit_lines), filenames)
                #true_sample = x.get(filenames + ''.join(str(sid) for sid in ref_id))
                sorted_x = sorted(x.items(), key=operator.itemgetter(0))
                count = 0
                for name, tf_idf in sorted_x:
                    if tf_idf > true_sample:
                        count = count +1
                print count
        raw_input()

    if not os.path.exists('stats/'):
        os.makedirs('stats/')
    table.save('stats/')
