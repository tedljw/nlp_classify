import codecs
import json
from .classify_processor import ClassifyProcessor

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_ifrn_examples(self, text_list):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with codecs.open(input_file, 'r', 'utf-8') as infs:
            for inf in infs:
                inf = inf.strip()
                dicts.append(json.loads(inf))
        return dicts

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, 'utf-8')
        lines = []
        for line in file_in:
            lines.append(line.split(","))
        return lines


def get_class(task_name):
    """ Mapping from task string to Dataset Class """
    processors = {"classify": ClassifyProcessor}
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    return processors[task_name]


