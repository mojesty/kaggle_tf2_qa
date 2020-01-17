local embedding_dim = 16;
local data_path = "./fixtures/simplified-nq-sample.jsonl";
{
  "dataset_reader": {
    "type": "natural_questions",
    "downsample_negative": -0.0,
    "downsample_all": 1.0,
  },
  "train_data_path": data_path,
  "validation_data_path": data_path,
  "model": {
    "type": "natural_questions",
    "word_embeddings": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embedding_dim,
        "trainable": true
      }
    },
    "encoder": {
        "type": "lstm",
        "input_size": embedding_dim,
        "hidden_size": 50,
        "num_layers": 1,
        "dropout": 0.0,
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
    "type": "basic",

  },
  "trainer": {
    "cuda_device": -1,
    "grad_norm": 5.0,
    // "validation_metric": "+f1-measure",
    "shuffle": false,
    "optimizer": {
      "type": "adam",
      "lr": 0.005
    },
    "num_epochs": 200,
    "patience": 200,
  }
}
