
import re
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

regex_punct = re.compile(r"[^ A-Za-z]+")
regex_proper_noun = re.compile(r"([^?.!])(\s+)([A-Z]\w*)")
regex_spaces = re.compile(r"\s+")

def check(string):
    return re.sub(regex_proper_noun, r"\1", string)

def fix_str(string):
    return re.sub(regex_spaces, " ", re.sub(regex_punct, "", re.sub(regex_proper_noun, r"\1", string)).lower())

def main():
    f = open("ggold.txt", "r")
    ctr = 0

    NUM_GOOD = None
    NUM_BAD = None

    good_strings = []
    bad_strings = []

    #doesn't matter, small loop.

    for line in f:
        if(ctr == 0):
            nums = line.split(' ')
            NUM_GOOD = int(nums[0])
            NUM_BAD = int(nums[1])
        elif(ctr < NUM_GOOD):
            good_strings.append(fix_str(line))
        else:
            bad_strings.append(fix_str(line))
        ctr += 1

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(good_strings)
    print(vectorizer.get_feature_names())
    
    model = KeyedVectors.load(get_tmpfile("glove100"), mmap='r')
