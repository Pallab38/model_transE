

def mrr(predictions, ground_truth_idx):
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    indices = predictions.argsort()
    mean_rank = (indices == ground_truth_idx).nonzero().float().add(1.0).sum().item()
    reciprocal_rank = 1.0/mean_rank

    return reciprocal_rank

def hit_at_k( predictions, ground_truth,k):
    zero_tensor = torch.Tensor([0])
    one_tensor = torch.Tensor([1])
    ground_truth = ground_truth#.cuda()
    _, indices = predictions.topk(k=k, largest=False)
    indices = indices.cuda()
    rank = torch.where(indices == ground_truth, one_tensor.cuda(), zero_tensor.cuda()).sum().item()

    return rank

def gen_mean_rank(predictions, ground_truth_idx):
    indices = predictions.argsort()
    mean_rank = (indices == ground_truth_idx).nonzero().float().add(1.0).sum().item()
    mean_rank =float(mean_rank / predictions.size()[0])

    return mean_rank

