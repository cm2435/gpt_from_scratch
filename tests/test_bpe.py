import unittest
import os
#from src.tokeniser.impl import Tokenizer
from tests.test_cases.tokenizer import test_strings
from tests.reference_implimentations.tokenizer import Tokenizer
def unpack(text):
    # we do this because `unittest` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__)) + "/test_cases"
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text


class TestBasicTokenizer(unittest.TestCase):

    def test_encode_decode_identity(self):
        for text in test_strings:
            text = unpack(text)
            tokenizer = Tokenizer(vocab_size=256 + 3)
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            self.assertEqual(text, decoded)

    def test_wikipedia_example(self):
        """
        Quick unit test, following along the Wikipedia example:
        https://en.wikipedia.org/wiki/Byte_pair_encoding

        According to Wikipedia, running bpe on the input string:
        "aaabdaaabac"

        for 3 merges will result in string:
        "XdXac"

        where:
        X=ZY
        Y=ab
        Z=aa

        Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
        so Z will be 256, Y will be 257, X will be 258.

        So we expect the output list of ids to be [258, 100, 258, 97, 99]
        """
        tokenizer = Tokenizer(vocab_size=256 + 3)
        text = "aaabdaaabac"
        tokenizer.train(text)
        ids = tokenizer.encode(text)
        self.assertEqual(ids, [258, 100, 258, 97, 99])
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

if __name__ == "__main__":
    unittest.main()