"""
Copyright 2019 Marcelo Silva

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import sys

class FreqTextSummarizer:

    def prep_text(self, text):
        return text.replace("\n", " ").replace("?", "").lower()

    def filter_stop_words(self, words, stop_words):
        return [word for word in words if word not in stop_words]

    def sent_rank(self, count):
        ids = nlargest(count, self.rank, key = self.rank.get)
        return [self.sents[id] for id in ids]

    def __init__(self, text, stop_words=None):
        prep_text = self.prep_text(text);
        words = self.filter_stop_words(word_tokenize(prep_text), stop_words)
        freq = FreqDist(words)
        self.rank = {}
        self.sents = sent_tokenize(prep_text)
        for id, sent in enumerate(self.sents):
            for word in word_tokenize(sent):
                if word in freq:
                    if word not in self.rank:
                        self.rank[id] = freq[word]
                    else:
                        self.rank[id] += freq[word]

if __name__ == "__main__":
    if len(sys.argv) is not 2:
        print("Usage: freqtextsummarizer filename")
        sys.exit()

    file_name = sys.argv[1]
    file = open(file_name, 'r')
    text = file.read()
    stop_words = set(stopwords.words('english') + list(punctuation))
    summ = FreqTextSummarizer(text, stop_words)
    print(summ.sent_rank(5))
