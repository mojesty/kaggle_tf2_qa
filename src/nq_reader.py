import itertools
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict

import jsonlines
from allennlp.common.checks import ConfigurationError

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
from allennlp.data.dataset_readers.dataset_utils.span_utils import iob1_tags_to_spans
from allennlp.data.fields import TextField, IndexField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter


@DatasetReader.register('natural_questions')
class NaturalQuestionsDatasetReader(DatasetReader):

    def __init__(self,
                 skip_empty: bool = False,
                 downsample_negative: float = 0.05,
                 simplified: bool = True,
                 skip_toplevel_answer_candidates: bool = True,
                 maxlen: int = 450,
                 classes_to_ignore: List[str] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True):
        if not simplified:
            raise ConfigurationError('Only simplified version of natural questions is allowed')
        super(NaturalQuestionsDatasetReader, self).__init__(lazy=lazy)
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._skip_empty = skip_empty
        self._maxlen = maxlen

        self._downsample_negative = downsample_negative
        self._skip_toplevel_answer_candidates = skip_toplevel_answer_candidates

    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(file_path) as reader:
            for inst in reader:
                # we tokenize text by space-splitting because this is
                # what is written in data description
                tokenized_text = inst['document_text'].split(' ')
                tokenized_question = inst['question_text'].split(' ')
                # candidates = inst['long_answer_candidates'] if not self._skip_toplevel_answer_candidates \
                #     else [cand for cand in inst['long_answer_candidates'] if not cand['top_level']]
                candidates = inst['long_answer_candidates']
                # there is only 1 annotation in train, so the list always has 0 element, sometimes
                # annotation may be absent (there is no answer in document)
                inst['annotations'] = inst['annotations'][0] if inst['annotations'] else None

                if inst['annotations'] is not None:
                    ann = inst['annotations']
                    correct_candidate_index = ann['long_answer']['candidate_index']

                    yes_no_answer = ann['yes_no_answer']
                    # if short answer is present, get it for squad-like QA
                    # if not, search for yes/no answer and construct target
                    # for classification
                    has_short_answer = len(ann['short_answers']) > 0
                    if not has_short_answer:
                        answer_start_token = answer_end_token = -1
                    else:
                        ann = ann['short_answers'][0]
                        start_token = inst['annotations']['long_answer']['start_token']
                        answer_start_token = ann['start_token'] - start_token
                        answer_end_token = min(
                            ann['end_token'] - start_token,
                            ann['start_token'] + self._maxlen - start_token
                        )

                    for idx, candidate in enumerate(candidates):
                        context = tokenized_text[candidate['start_token']:candidate['end_token']][:self._maxlen]
                        if idx != correct_candidate_index and random.uniform(0, 1) < self._downsample_negative:
                            # negative example, we downsample it first
                            categorical_target = 'not_relevant'
                            yield self.text_to_instance(
                                context,
                                tokenized_question,
                                categorical_target,
                                -1,
                                -1,
                                example_id=inst['example_id']
                            )
                        else:
                            # positive example,
                            categorical_target = yes_no_answer
                            yield self.text_to_instance(
                                context,
                                tokenized_question,
                                categorical_target,
                                answer_start_token,
                                answer_end_token,
                                example_id=inst['example_id']
                            )

    def text_to_instance(self,
                         tokens: List[str],
                         query: List[str],
                         categorical_target: str = None,
                         answer_start_token: int = None,
                         answer_end_token: int = None,
                         **metadata
                         ) -> Instance:
        """

        :param tokens: tokens
        :param query: question
        :param answer_start_token: token index, may be -1 if answer is not relevant
        :param answer_end_token: token index, may be -1 if answer is not relevant
        :param metadata: metadata about instance, e.g. example_id or document_url.
            passed to model and to predictor to make evaluation simpler
        :return:
        """
        if answer_start_token is None and answer_end_token is not None:
            raise RuntimeError(f'start and end tokens must be simultaneously None, got {answer_start_token} and {answer_end_token}')
        if categorical_target is None and answer_start_token is not None:
            raise RuntimeError(f'categorical target and answer start token must be simultaneously None')
        if not tokens:
            raise RuntimeError(f'Empty context passed to natural questions instance generator')
        if not query:
            raise RuntimeError(f'Empty query passed to natural questions instance generator')
        text_and_query_field = TextField(
            [Token('[CLS]')] + [Token(word) for word in query]
            + [Token('[SEP]')] + [Token(word) for word in tokens],
            self._token_indexers
        )
        metadata.update({'text': tokens, 'query': query})
        fields = {
            'context': text_and_query_field,
            'meta': MetadataField(metadata)
        }
        # fields['answer_starts'] = IndexField(span_start, passage_field)
        if answer_start_token is not None and answer_end_token is not None:
            fields['answer_start'] = IndexField(answer_start_token, sequence_field=text_and_query_field)
            fields['answer_end'] = IndexField(answer_end_token, sequence_field=text_and_query_field)
            fields['answer_label'] = LabelField(categorical_target)
        return Instance(fields)


        # sentences = Path(file_path).read_text().split('\n\n')
        # for sentence in sentences:
        #     tokens = []
        #     targets = []
        #     for token_line in sentence.split('\n'):
        #         text, *labels = token_line.split('\t')
        #         tokens.append(text)
        #         targets.append(labels[self._target_column - 1])
        #     entities_spans: List[TypedStringSpan] = iob1_tags_to_spans(targets)
        #     # one sentence may have several entities of the same type, so it is not
        #     # exactly squad-like task, where only 1 span may be answer
        #     # to take this into account, we group spans by their type
        #     # and for each _group_ create an instance with _multiple_
        #     # answer spans. The model learns to predict start and end token
        #     # for _each_ of these spans.
        #     entities_spans_grouped = itertools.groupby(sorted(entities_spans, key=lambda x: x[0]), key=lambda x: x[0])
        #     for key, group in entities_spans_grouped:
        #         spans = list(group)
        #         query = self._descriptions[self._context_type][key]
        #         query = query.split(' ')  # TODO (yaroslav): make this better 27.12.19
        #         answer_starts: List[int] = [span[1][0] for span in spans]
        #         answer_ends: List[int] = [span[1][1] for span in spans]
        #         yield self.text_to_instance(tokens, query, answer_starts, answer_ends, type=key)
