def calc_mean_rank(rank):
    mean_rank = 0
    N = len(rank)
    print("Length of rank: ", N)
    for i in rank:
        mean_rank = mean_rank + i / N

    return mean_rank

def calc_mean_raw_rank(rank):
    mean_rank_raw = 0
    N = len(rank)
    print("Length of mean rank : ", N)
    for i in rank:
        mean_rank_raw = mean_rank_raw +1 /i / N

    return mean_rank_raw

def calc_hit_rank_N(rank, N):
    hit_rank = 0
    for i in rank:
        if i <= N:
            hit_rank = hit_rank + 1

    hit_rank = hit_rank / len(rank)

    return hit_rank
