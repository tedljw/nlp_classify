from utils.classify_utils import convert_examples_to_features, convert_features_to_tensors
from sklearn import metrics
import numpy as np
import torch

def evaluate(model, processor, args, label_list, tokenizer, device):
    '''模型验证

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, show_exp=False)

    eval_dataloader = convert_features_to_tensors(eval_features, args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            #pred = np.argmax(logits, axis=1)
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        #logits = logits.detach().cpu().numpy()
        #label_ids = label_ids.to('cpu').numpy()

    print(len(gt))
    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print(f1)

    return f1


def test(model, processor, args, label_list, tokenizer, device):
    '''模型测试

    Args:
        model: 模型
	processor: 数据读取方法
	args: 参数表
	label_list: 所有可能类别
	tokenizer: 分词方法
	device

    Returns:
        f1: F1值
    '''
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)

    test_dataloader = convert_features_to_tensors(test_features, args.eval_batch_size)

    model.eval()
    predict = np.zeros((0,), dtype=np.int32)
    gt = np.zeros((0,), dtype=np.int32)
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            pred = logits.max(1)[1]
            predict = np.hstack((predict, pred.cpu().numpy()))
            gt = np.hstack((gt, label_ids.cpu().numpy()))

        #logits = logits.detach().cpu().numpy()
        #label_ids = label_ids.to('cpu').numpy()

    f1 = np.mean(metrics.f1_score(predict, gt, average=None))
    print('F1 score in text set is {}'.format(f1))

    return f1