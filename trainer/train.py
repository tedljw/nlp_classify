from config.config_processor import read_config
import utils.utils as utils
from processor.data_processor import get_class
from main import logger
from utils.classify_utils import convert_examples_to_features, convert_features_to_tensors
from .evaluate_test import evaluate, test

from tqdm import tqdm, trange

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

"""
    本代码参考pytorch-pretrained-BERT的run_classifier.py，
    但是删掉了GPU分布式训练和16位浮点精度的功能，
    所以本代码目前仅适用于单显卡的单机版
"""
def train(config):
    cfg, cfg_data, cfg_model, cfg_optim = read_config(config)

    device, n_gpu = utils.get_device()
    utils.set_seeds(cfg.seed, n_gpu)

    train_batch_size = int(cfg_optim.train_batch_size / cfg_optim.gradient_accumulation_steps)

    processor = get_class(cfg.task.lower())

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model, do_lower_case=cfg.do_lower_case)

    train_examples = None
    num_train_steps = None
    if cfg.do_train:
        train_examples = processor.get_train_examples(cfg_data.data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size / cfg_optim.gradient_accumulation_steps * cfg_optim.num_train_epochs)

    label_list = processor.get_labels()
    # Prepare model
    print(PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format( -1 ) )
    model = BertForSequenceClassification.from_pretrained(cfg.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              -1 ), num_labels=len(label_list))

    model.to(device)

    # Prepare optimizer
    if cfg_optim.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=cfg_optim.learning_rate,
                         warmup=cfg_optim.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if cfg.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, cfg_optim.max_seq_length, tokenizer, show_exp=False)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = convert_features_to_tensors(train_features, train_batch_size)

        model.train()
        best_score = 0
        flags = 0
        for _ in trange(int(cfg_optim.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if cfg_optim.fp16 and cfg_optim.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * cfg_optim.loss_scale
                if cfg_optim.gradient_accumulation_steps > 1:
                    loss = loss / cfg_optim.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % cfg_optim.gradient_accumulation_steps == 0:
                    if cfg_optim.optimize_on_cpu:
                        if cfg_optim.fp16 and cfg_optim.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / cfg_optim.loss_scale
                        is_nan = utils.set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            cfg_optim.loss_scale = cfg_optim.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        utils.copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()

            f1 = evaluate(model, processor, cfg_optim, label_list, tokenizer, device)
            if f1 > best_score:
                best_score = f1
                print('*f1 score = {}'.format(f1))
                flags = 0
                checkpoint = {
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, cfg_optim.model_save_pth)
            else:
                print('f1 score = {}'.format(f1))
                flags += 1
                if flags >= 6:
                    break

    model.load_state_dict(torch.load(cfg.model_save_pth)['state_dict'])
    test(model, processor, cfg_optim, label_list, tokenizer, device)