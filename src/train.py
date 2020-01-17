import os
import sys

from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules

sys.path.append(os.path.abspath('..'))

if __name__ == '__main__':

    import_submodules('src')
    train_model_from_file(
        'natural-questions-simplified-full.jsonnet',
        '/home/emelyanov-yi/models/tf2_qa/init3',
        force=True
    )
