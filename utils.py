from random import randint


def gen_data(low, high, n_rows, n_cols=None):
    '''
    随机生成数据集
    :param low: {int} -- 生成元素的最小值
    :param high: {int} -- 生成元素的最大值
    :param n_rows: {int} -- 行数
    :param n_cols: {int} -- 列数
    :return: {list} -- int型1维或者2维列表
    '''
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret


def get_eu_dist(arr1: list, arr2: list) -> float:
    '''
    计算两个向量间的欧式距离
    :param arr1: {list} -- int或者float类型的1维列表对象
    :param arr2: {list} -- int或者float类型的1维列表对象
    :return: {float} -- 欧式距离
    '''
    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5
