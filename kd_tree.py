from utils import get_eu_dist

class Node(object):
    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None

    def __str__(self):
        return "feature: %s, split: %s" % (str(self.feature), str(self.split))

    @property
    def brother(self):
        '''
        找到兄弟节点
        :return: 兄弟节点
        '''
        if not self.father:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret


class KDTree(object):
    def __init__(self):
        # 构造根节点
        self.root = Node()

    def __str__(self):
        '''
        展示Kd树中每个节点之前的关系
        :return: Kd树中节点的信息
        '''
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            # node, 父节点索引
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))

            if nd.left:
                que.append((nd.left, i))
            if nd.right:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)

    def _get_median_idx(self, X, idxs, feature):
        '''
        计算一列数据的中位数
        :param X: {list} -- int或者float类型的2维列表对象
        :param idxs: {list} -- int类型的1维列表
        :param feature: {int} -- 特征数
        :return: {list} -- 对应于这一列中值的行索引
        '''
        n = len(idxs)

        # Ignoring the number of column elements is odd and even.
        k = n // 2

        # Get all the indexes and elements of column j as tuples.
        col = map(lambda i: (i, X[i][feature]), idxs)

        # Sort the tuples by the elements' values
        # and get the corresponding indexes.
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))

        # Search the median value.
        median_idx = list(sorted_idxs)[k]
        return median_idx

    def _get_variance(self, X, idxs, feature):
        '''
        计算一列数据的方差
        :param X: {list} -- int或者float类型的2维列表对象
        :param idxs: {list} -- int类型的1维列表
        :param feature: {int} -- 特征数
        :return: {float} -- 方差
        '''
        n = len(idxs)
        col_sum = col_sum_sqr = 0
        for idx in idxs:
            xi = X[idx][feature]
            col_sum += xi
            col_sum_sqr += xi ** 2
        # D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        return col_sum_sqr / n - (col_sum / n) ** 2

    def _choose_feature(self, X, idxs):
        '''
        选择具有方差最大的特征
        :param X: {list} -- int或者float类型的2维list对象
        :param idxs: {list} -- int类型的1维列表
        :return: {int} -- 特征数
        '''
        m = len(X[0])
        variances = map(lambda j: (
            j, self._get_variance(X, idxs, j)), range(m))
        return max(variances, key=lambda x: x[1])[0]

    def _split_feature(self, X, idxs, feature, median_idx):
        '''
        根据拆分点将索引拆分为两个数组
        :param X: {list} -- int或者float类型的2维list对象
        :param idxs: {list} -- 索引, int类型的1维列表对象
        :param feature: {int} -- 特征数
        :param median_idx: {float} -- 特征的中间索引
        :return: {list} -- [left idx, right idx]
        '''
        idxs_split = [[], []]
        split_val = X[median_idx][feature]
        for idx in idxs:
            # Keep the split point in current node.
            if idx == median_idx:
                continue

            # Split
            xi = X[idx][feature]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

    def build_tree(self, X, y):
        '''
        创建Kd树。数据应按比例计算，以便计算方差
        :param X: {list} -- int或者float类型的2维列表对象
        :param y: {list} -- int或者float类型的1维列表对象
        :return:
        '''

        # Initialize with node, indexes
        nd = self.root
        idxs = range(len(X))
        que = [(nd, idxs)]
        while que:
            nd, idxs = que.pop(0)
            n = len(idxs)

            # Stop split if there is only one element in this node
            if n == 1:
                nd.split = (X[idxs[0]], y[idxs[0]])
                continue

            # Split
            feature = self._choose_feature(X, idxs)
            median_idx = self._get_median_idx(X, idxs, feature)
            idxs_left, idxs_right = self._split_feature(
                X, idxs, feature, median_idx)

            # Update properties of current node
            nd.feature = feature
            nd.split = (X[median_idx], y[median_idx])

            # Put children of current node in que
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                que.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                que.append((nd.right, idxs_right))

    def _search(self, Xi, nd):
        '''
        在Kd树中查找Xi，直到Xi在叶子结点上
        :param Xi: {list} -- int或者float类型的1维列表
        :param nd: node，某个节点
        :return: 叶子节点
        '''
        while nd.left or nd.right:
            if not nd.left:
                nd = nd.right
            elif not nd.right:
                nd = nd.left
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    def _get_eu_dist(self, Xi, nd):
        '''
        计算节点Xi和node之间的欧式距离
        :param Xi: {list} -- int或者float类型的1维列表
        :param nd: 某个节点
        :return: {float} -- 欧式距离
        '''
        X0 = nd.split[0]
        return get_eu_dist(Xi, X0)

    def _get_hyper_plane_dist(self, Xi, nd):
        '''
        计算超平面距离
        :param Xi: {list} -- int或者float类型的1维列表
        :param nd: {node} -- 某个节点
        :return: {float} -- 欧式距离
        '''
        j = nd.feature
        X0 = nd.split[0]
        return abs(Xi[j] - X0[j])

    def nearest_neighbour_search(self, Xi):
        '''
        KNN的查找和回溯
        :param Xi: {list} -- int或者float类型的1维列表
        :return: {node} -- 离Xi最近的节点
        '''

        # The leaf node after searching Xi.
        dist_best = float("inf")
        nd_best = self._search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur = que.pop(0)

            # Calculate distance between Xi and root node
            dist = self._get_eu_dist(Xi, nd_root)

            # Update best node and distance.
            if dist < dist_best:
                dist_best, nd_best = dist, nd_root
            while nd_cur is not nd_root:

                # Calculate distance between Xi and current node
                dist = self._get_eu_dist(Xi, nd_cur)

                # Update best node, distance and visit flag.
                if dist < dist_best:
                    dist_best, nd_best = dist, nd_cur

                # If it's necessary to visit brother node.
                if nd_cur.brother and dist_best > \
                        self._get_hyper_plane_dist(Xi, nd_cur.father):
                    _nd_best = self._search(Xi, nd_cur.brother)
                    que.append((nd_cur.brother, _nd_best))

                # Back track.
                nd_cur = nd_cur.father

        return nd_best