import os
from processor.data_processor import DataProcessor
from utils.classify_utils import InputExample

class ClassifyProcessor(DataProcessor):
	"""Processor for the MRPC data set (GLUE version)."""

	def __init__(self):
		self.labels = set()

	def get_train_examples(self, data_dir):
		return self._create_examples(
			self._read_json(os.path.join(data_dir, "train.json")), 'train')

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self._read_json(os.path.join(data_dir, "valid.json")), 'dev')

	def get_test_examples(self, data_dir):
		return self._create_examples(
			self._read_json(os.path.join(data_dir, "test.json")), 'test')

	def get_ifrn_examples(self, text_list):
		return self._create_ifrn_examples(text_list, "ifrn")

	def get_labels(self):
		"""See base class."""
		return list(self.labels)

	def _create_ifrn_examples(self, text_list, set_type):
		examples = []
		#        for (i, text) in enumerate(text_list):

		guid = "%s-%s" % (set_type, 1)
		examples.append(
			InputExample(guid=guid, text_a=text_list))
		return examples

	def _create_examples(self, dicts, set_type):
		examples = []
		for (i, infor) in enumerate(dicts):
			guid = "%s-%s" % (set_type, i)
			text_a = infor['question']
			label = infor['label']
			self.labels.add(label)
			examples.append(
				InputExample(guid=guid, text_a=text_a, label=label))
		return examples