import numpy as np
import warnings
from collections import defaultdict
from IPython import embed
from scipy import stats

try:
    from lreid.evaluation.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    
    # embed()
    # compute 95% confidence interval of Rank1
    rank1_error1 = stats.binom.interval(0.95, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    rank1_error2 = stats.binom.interval(0.975, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    re1 = np.mean(abs(rank1_error1-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])
    re2 = np.mean(abs(rank1_error2-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])

    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    
    # compute 95% confidence interval of mAP
    mAP_error1 = stats.t.interval(0.95,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    mAP_error2 = stats.t.interval(0.975,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    alpha = np.mean(abs(mAP_error1-mAP))
    beta = np.mean(abs(mAP_error2-mAP))

    print("95% mAP Error is {}, 95% Rank1 Error is {}".format(alpha,re1))
    print("97.5% mAP Error is {}, 97.5% Rank1 Error is {}".format(beta,re2))

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1) # 每行,从小到大排序(一个query的多个检索结果,相似性从小到大排序)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx] #每一个query的排序列表
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove) # 取反
        # print("rank.py====================line124")
        # embed()


        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum() # 求和
        tmp_cmc = raw_cmc.cumsum() # 累积和
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)


    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    

    
    all_cmc = np.asarray(all_cmc).astype(np.float32) # (3368, 50)
    embed()
    # compute 95% confidence interval of Rank1
    rank1_error1 = stats.binom.interval(0.95, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    rank1_error2 = stats.binom.interval(0.975, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    re1 = np.mean(abs(rank1_error1-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])
    re2 = np.mean(abs(rank1_error2-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])

    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    
    # compute 95% confidence interval of mAP
    mAP_error1 = stats.t.interval(0.95,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    mAP_error2 = stats.t.interval(0.975,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    alpha = np.mean(abs(mAP_error1-mAP))
    beta = np.mean(abs(mAP_error2-mAP))

    print("95% mAP Error is {}, 95% Rank1 Error is {}".format(alpha,re1))
    print("97.5% mAP Error is {}, 97.5% Rank1 Error is {}".format(beta,re2))
    # print("--------rank.py----------line 156")
    # embed()

    return all_cmc, mAP


def eval_cuhksysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)
        keep = np.ones(len(order), dtype=bool) #全是true

      
        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    
    # compute 95% confidence interval of Rank1
    rank1_error1 = stats.binom.interval(0.95, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    rank1_error2 = stats.binom.interval(0.975, num_valid_q, all_cmc.sum(0)[0]/num_valid_q)
    re1 = np.mean(abs(rank1_error1-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])
    re2 = np.mean(abs(rank1_error2-all_cmc.sum(0)[0])/all_cmc.sum(0)[0])

    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    
    # compute 95% confidence interval of mAP
    mAP_error1 = stats.t.interval(0.95,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    mAP_error2 = stats.t.interval(0.975,len(np.array(all_AP))-1,loc = mAP, scale=stats.sem(np.array(all_AP)))
    alpha = np.mean(abs(mAP_error1-mAP))
    beta = np.mean(abs(mAP_error2-mAP))

    print("95% mAP Error is {}, 95% Rank1 Error is {}".format(alpha,re1))
    print("97.5% mAP Error is {}, 97.5% Rank1 Error is {}".format(beta,re2))

    return all_cmc, mAP



def evaluate_py(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )
    else:
        return eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )

def evaluate_py_cuhksysu(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )
    else:
        return eval_cuhksysu(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )

def fast_evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    use_metric_cuhk03=False,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    # from IPython import embed
    # embed()
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
    else:
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )

def fast_evaluate_rank_cuhksysu(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    use_metric_cuhk03=False,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
    else:
        return evaluate_py_cuhksysu(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )