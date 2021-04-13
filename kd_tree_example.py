from kd_tree import KDTree
from utils import gen_data
from utils import get_eu_dist
from time import time


def exhausted_search(X, Xi):
    '''
    线性查找KNN
    :param X: {list} -- int或者float类型的2维列表
    :param Xi: {list} -- int或者float类型的1维列表
    :return: {list} -- int或者float类型的1维列表
    '''
    dist_best = float('inf')
    row_best = None
    for row in X:
        dist = get_eu_dist(Xi, row)
        if dist < dist_best:
            dist_best = dist
            row_best = row
    return row_best


def main():
    print("Testing KD Tree...")
    test_times = 100
    run_time_1 = run_time_2 = 0

    for _ in range(test_times):
        # 随机生成数据
        low = 0
        high = 100
        n_rows = 1000
        n_cols = 2
        X = gen_data(low, high, n_rows, n_cols)
        y = gen_data(low, high, n_rows)
        Xi = gen_data(low, high, n_cols)

        # 创建Kd树
        tree = KDTree()
        tree.build_tree(X, y)

        # Kd树查找
        start = time()
        nd = tree.nearest_neighbour_search(Xi)
        run_time_1 += time() - start
        ret1 = get_eu_dist(Xi, nd.split[0])

        # 普通线性查找
        start = time()
        row = exhausted_search(X, Xi)
        run_time_2 += time() - start
        ret2 = get_eu_dist(Xi, row)

        # 比较结果
        assert ret1 == ret2, "target:%s\nrestult1:%s\nrestult2:%s\ntree:\n%s" % (Xi, nd, row, tree)

    print("%d tests passed!" % test_times)
    print("KD Tree Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


if __name__ == "__main__":
    main()
