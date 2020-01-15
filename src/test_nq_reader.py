import unittest

from allennlp.common.util import ensure_list
from allennlp.data import Token

from src.nq_reader import NaturalQuestionsDatasetReader


class TestNerQAReader(unittest.TestCase):

    def test_01_read_file(self):
        reader = NaturalQuestionsDatasetReader(
            downsample_negative=1.0,
        )

        instances = reader.read('fixtures/simplified-nq-sample.jsonl')
        instances = ensure_list(instances)
        self.assertEqual(len(instances), 1125)
        self.assertEqual(instances[53].fields['answer_label'].label, 'not_relevant')
        self.assertEqual(instances[54].fields['answer_label'].label, 'NONE')
        # checking 1st instance
        # inst = instances[0]
        # self.assertEqual(sum(inst['answer_starts']), 1)
        # self.assertIn(Token('[SEP]'), inst['context'].tokens, 'context is not well-formed "[CLS] query [SEP] text" question')
        # self.assertEqual(inst['context'].tokens.index(Token('nationalities')), 1)
        # self.assertEqual(inst['answer_starts'], inst['answer_ends'])
        # self.assertEqual(inst['answer_starts'].labels.index(True), 14)
        # self.assertEqual(inst['meta'].metadata['type'], 'NORP')
        # # 3rd and 4th instances must have same context but different queries
        # self.assertEqual(instances[2]['meta'].metadata['text'], instances[3]['meta'].metadata['text'])
        # self.assertNotEqual(instances[2]['meta'].metadata['query'], instances[3]['meta'].metadata['query'])


if __name__ == '__main__':
    unittest.main()
