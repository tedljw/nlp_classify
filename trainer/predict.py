from config.config_processor import read_config
import utils.utils as utils
from processor.data_processor import get_class
from utils.classify_utils import convert_examples_to_features, convert_features_to_tensors

import time
import torch
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

return_text = True

def init_model(config):
    cfg, cfg_data, cfg_model, cfg_optim = read_config(config)

    device, n_gpu = utils.get_device()
    utils.set_seeds(cfg.seed, n_gpu)

    train_batch_size = int(cfg_optim.train_batch_size / cfg_optim.gradient_accumulation_steps)

    processor = get_class(cfg.task.lower())

    processor.get_train_examples(cfg.data_dir)

    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model, do_lower_case=cfg.do_lower_case)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              -1), num_labels=len(label_list))

    model.to(device)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(cfg.model_save_pth, map_location='cpu')['state_dict'])
    else:
        model.load_state_dict(torch.load(cfg.model_save_pth)['state_dict'])

    return model, processor, cfg_optim, label_list, tokenizer, device



class PredictProcessor :
    def __init__(self, model, processor, args, label_list, tokenizer, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    def predict(self, text_list):
        result = []
        print(text_list)
        test_examples = self.processor.get_ifrn_examples(text_list)
        print("test_examples", test_examples[0].text_a)

        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.args.max_seq_length, self.tokenizer, show_exp=False)

        test_dataloader = convert_features_to_tensors(test_features, batch_size=self.args.eval_batch_size)

        for idx, (input_ids, input_mask, segment_ids) in enumerate(test_dataloader):
            item = {}
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            text = test_examples[idx].text_a
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = F.softmax(logits, dim=1)
                pred = logits.max(1)[1]
                #logits = logits.detach().cpu().numpy()[0].tolist()
                if return_text:
                    item['text'] = text
                item['label'] = pred.item()
                item['scores'] = {0: logits[item['label']]}
                result.append(item)
        return result


def predict(config):

    print('[INFO]Init model started.')
    model, processor, cfg_optim, label_list, tokenizer, device = init_model(config)
    print('[INFO]Init model finished.')
    ph = PredictProcessor(model, processor, cfg_optim, label_list, tokenizer, device)

    start_time = time.clock()

    result = ph.predict("我想知道我买的蓝蜂手柄到底如何使用")
    stop_time = time.clock()
    cost = stop_time - start_time
    print("Predict cost %s second" % cost)
    for res in result:
        print(res)
