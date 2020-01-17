# pylint: disable=invalid-name,protected-access
import json
import unittest
from copy import deepcopy

from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from src import NaturalQuestionsModel, NaturalQuestionsDatasetReader


class SimpleRETest(ModelTestCase):
    # FIXTURES_ROOT = PROJECT_ROOT / "components" / "tests" / "fixtures"

    def setUp(self):
        super().setUp()
        self.set_up_model(
            "fixtures/natural-questions-simplified.jsonnet",
            "./fixtures/simplified-nq-sample.jsonl",
        )

    def test_01_simple_qa_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_01_simple_qa_can_train_save_and_load_stochastically(self):
        overrides = json.dumps({
            'dataset_reader.downsample_negative': 0.1
        })
        self.ensure_model_can_train_save_and_load(self.param_file, overrides=overrides)

    @flaky
    @pytest.mark.skip
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        logits = output_dict["logits"]
        assert logits.shape[-1] == 2
        # assert len(logits[0]) == 7
        # assert len(logits[1]) == 7
        # for example_tags in logits:
        #     for tag_id in example_tags:
        #         tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
        #         assert tag in {'0', '1'}

