# Copyright 2019 Marcelo Silva

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import sys

class SentRanker:
    """
        This class creates a rank for the sentences in a given text. The rank
        uses the word frenquency to give higher ranks to sentences containing
        more frequent words.
    """
    def _prep_text(self, text):
        return text.replace("\n", " ").replace("?", "").lower()

    def _filter_stop_words(self, words, stop_words):
        return [word for word in words if word not in stop_words]

    def sent_rank(self, count):
        """
            Returns a list of ranked sentences which are sorted in decreasing
            order of rank. The number of sentences returned is given by count.
            The first sentence in the list has the highest rank.

            Parameters:
            count (int): The number of sentences to be returned

            Returns:
            int: A list of ranked sentences.
        """
        ids = nlargest(count, self.rank, key=self.rank.get)
        return [self.sents[id] for id in ids]

    def __init__(self, text, stop_words):
        """
            Creates a SentRanker instance for the given text and stop words.

            Parameters:
            text (string): The text
            stop_words (set): The stop words
        """
        prep_text = self._prep_text(text);
        words = self._filter_stop_words(word_tokenize(prep_text), stop_words)
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

# main
if __name__ == "__main__":
    if len(sys.argv) is not 2:
        print("Usage: sent_rank.py filename")
        sys.exit()

    file_name = sys.argv[1]
    file = open(file_name, 'r')
    text = file.read()
    file.close()
    stop_words = set(stopwords.words('english') + list(punctuation))
    ranker = SentRanker(text, stop_words)

    # Prints the 5 highest ranked sentences
    print(ranker.sent_rank(5))
