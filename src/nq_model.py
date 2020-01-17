from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from torch import nn, Tensor

# from components.data.metrics.instance_wise_accuracy import InstanceWiseCategoricalAccuracy
from torch.nn import CrossEntropyLoss


TensorDict = Dict[str, torch.Tensor]

# TODO: support nested entities
@Model.register("natural_questions")
class NaturalQuestionsModel(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary,

        metrics: Dict[str, Metric] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        hidden_dim = self.encoder.get_output_dim()
        self.span_starts = nn.Linear(hidden_dim, 1)
        self.span_ends = nn.Linear(hidden_dim, 1)
        self._target_namespace = 'answer_labels'
        self.label_head = nn.Sequential(
            nn.Linear(encoder.get_output_dim(), 32),
            nn.Dropout(0.3),
            nn.Linear(32, self.vocab.get_vocab_size(self._target_namespace))
        )
        self._label_loss_fn = CrossEntropyLoss()
        self._qa_loss_fn = CrossEntropyLoss()
        metrics = metrics or {}
        metrics.update({
            "label_accuracy": CategoricalAccuracy(),
            "start_accuracy": CategoricalAccuracy(),
            "end_accuracy": CategoricalAccuracy(),
            "label_f1measure": FBetaMeasure(),
        })
        self.metrics = metrics
        initializer(self)

    def forward(
        self,
        context: TensorDict,  # [B , L , N]
        answer_label: Tensor = None,  # [B]
        answer_start: Tensor = None,  # [B , L]
        answer_end: Tensor = None,  # [B , L]
        **kwargs,
    ) -> Dict:
        # B -- batch_size
        # L -- number of lines in batch
        # N -- number of tokens in line
        # E -- token embedding dim

        if answer_start is None and answer_end is not None:
            raise RuntimeError(f'Answer start ans answer ends must be provided simultaneously')
        if answer_end is None and answer_start is not None:
            raise RuntimeError(f'Answer start ans answer ends must be provided simultaneously')
        mask = get_text_field_mask(context)  # [B , L]
        embeddings = self.word_embeddings(context)  # [B , N , E]
        batch_size = embeddings.size(0)
        num_tokens = embeddings.size(1)
        encoded_lines = self.encoder(embeddings, mask=mask)
        # encoded lines has embeddings for [CLS] token and other tokens (query, [SEP] and context)
        # we use different heads for them, so split this tensor to route its different parts
        # for their respective heads
        cls_embeddings, tokens_embeddings = encoded_lines.split([1, encoded_lines.size(1)-1], dim=1)
        # cls_embeddings do not need spatial (or token-wise) dimension as they are
        # intended for simple classification task
        cls_embeddings.squeeze_(1)
        start_logits = self.span_starts(tokens_embeddings)  # [B , N , 1]
        end_logits = self.span_ends(tokens_embeddings)  # [B , N , 1]
        start_logits = start_logits.squeeze(-1)  # [B , N]
        end_logits = end_logits.squeeze(-1)  # [B , N]

        label_logits = self.label_head(cls_embeddings)
        output = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            'label_logits': label_logits,
            "mask": mask,
            **kwargs
        }

        if answer_start is not None and answer_label is not None:

            # for metric in self.metrics.values():
            self.metrics['label_accuracy'](label_logits, answer_label)
            self.metrics["label_f1measure"](label_logits, answer_label)

            answer_start.clamp_(max=num_tokens - 2)
            answer_end.clamp_(max=num_tokens - 2)
            self.metrics['start_accuracy'](start_logits, answer_start.flatten())
            self.metrics['end_accuracy'](end_logits, answer_end.flatten())
            # self.metrics['span_f1'](start_logits, end_logits, answer_start, answer_end, [meta['type'] for meta in kwargs['meta']])
            label_loss = self._label_loss_fn(label_logits, answer_label)
            start_loss = self._qa_loss_fn(start_logits, answer_start.flatten())
            end_loss = self._qa_loss_fn(end_logits, answer_end.flatten())
            loss = label_loss / 10 + start_loss + end_loss
            output['loss'] = loss
            output['label_loss'] = label_loss
            output['qa_loss'] = start_loss + end_loss
        return output

    # def decode(self, output_dict: Dict[str, torch.Tensor]):
    #     """Get argmax over logits and convert to tags"""
    #     logits = output_dict["logits"]
    #     argmax = logits.argmax(dim=-1)
    #     tags = [
    #         self.vocab.get_token_from_index(idx.item(), self._target_namespace)
    #         for idx in argmax
    #     ]
    #     output_dict["tags"] = tags
    #     return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        for metric_name, metric in self.metrics.items():
            metrics = metric.get_metric(reset)
            if isinstance(metrics, float):
                metrics_to_return[metric_name] = metrics
            elif isinstance(metrics, tuple) and len(metrics) == 3:
                # handle case of fmeasure (for example)
                metrics_to_return["precision"], metrics_to_return[
                    "recall"
                ], metrics_to_return["f1-measure"] = metrics
            elif isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, float):
                        # simple case that handles averaged f measure
                        metrics_to_return[k] = v
                    elif isinstance(v, list):
                        # when f measure is not averaged
                        for i, m in enumerate(v):
                            label = self.vocab.get_token_from_index(
                                i, self._target_namespace
                            )
                            metrics_to_return[k + "_" + label] = m

                # metrics_to_return.update(metrics)
            else:
                raise RuntimeError(f"Metric {metric_name} cannot be displayed")
        return metrics_to_return
