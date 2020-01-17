local embedding_dim = 16;
local data_prefix = "/home/emelyanov-yi/Downloads";
local bertname='bert-base-cased';
{
  "dataset_reader": {
    "type": "natural_questions",
    "downsample_negative": 0.05,
    "downsample_all": 0.01,
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": bertname,
        "use_starting_offsets": true,
        "do_lowercase": false
      },
    }
  },
  "train_data_path": data_prefix + '/simplified-nq-train-trunc.jsonl',
  "validation_data_path": data_prefix + '/simplified-nq-dev.jsonl',
  "model": {
    "type": "natural_questions",
    "word_embeddings": {
      "token_embedders": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": bertname,
            "top_layer_only": false,
            "requires_grad": false,
          },
        },
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"]
        },
        "allow_unmatched_keys": true
    },
    "encoder": {
        "type": "lstm",
        "input_size": 768,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": true
    },
//    "label_head": {
//        "input_dim": 50 * 2,
//        "num_layers": 2,
//        "hidden_dims": [64],
//        "dropout": 0.3
//    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["context", "num_tokens"]],
    "biggest_batch_first": true,
  },
  "trainer": {
    "cuda_device": 0,
    "grad_norm": 5.0,
    // "validation_metric": "+f1-measure",
    "shuffle": false,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "num_epochs": 100,
    "patience": 20,
  }
}
