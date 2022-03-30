import numpy as np
import time
import itertools
import pickle

from anytree import AnyNode, NodeMixin, RenderTree

'''
    #######- GREEDY ALGORITHM -#######
    A public item must be suppressed, if the item on ITS OWN IS A MOLE.
        If a public item is a (size-1) mole, the item will
        not occur in any (h,k,p)-cohesion of D, thus, can be suppressed in
        a preprocessing step


'''


def find_subarrays_of_size_n(l: list, n: int):
    """

    :param l: The list of elements
    :param n: Number of elements in a sub-array
    :return: A map object containing the n-length sub-arrays
    """
    return itertools.combinations(l, n)


def find_unique_items(input_file):
    with open(input_file, "r") as file:
        lines = file.read()
        unique_items = set(lines.split())

        return unique_items


# FIXME: Score table could be a dict
# class ScoreTable:
#     def __init__(self, ):
class Node(NodeMixin):
    def __init__(self, label=None, mole_num=None, node_link=None, parent=None, children=None):
        """

        :param label: The item at this node
        :param mole_num: The number of minimal moles pass this node
        :param node_link: The link pointing to the next node with the same label
        :param parent: The parent node of this node
        :param children: Children nodes if any
        """
        super(Node, self).__init__()
        self.label = label
        self.mole_num = mole_num
        self.node_link = node_link
        self.parent = parent
        if children:
            self.children = children


# this can be an inner class of hkp coherence
class Transaction:
    # tag the private and pub items of each transaction in data
    def __init__(self, ID: int, public: list, private: list):
        self.ID = ID
        self.public = public
        self.private = private


class HKPCoherence:
    # TODO: should I save the suppressed items into a global variable
    def __init__(self, dataset: list, public_item_list: list, private_item_list: list, h: float, k: int, p: int):
        """

        :param dataset: The list of transactions
        :param public_item_list: Public items of the dataset
        :param private_item_list: Private items of the dataset
        :param h: The percentage of the transactions in beta-cohort that contain a common private item
        :param k: The least number of transactions that should be contained in the beta-cohort
        :param p: The maximum number of public items that can be obtained as prior knowledge in a single attack
        """
        # includes all private and public items
        self.dataset = dataset
        self.public_item_list = public_item_list  # list of public items among all data
        self.private_item_list = private_item_list  # list of priv items among all data
        self.h = h
        self.k = k
        self.p = p
        self.transactions = list()
        self.size1_moles = list()
        self.moles = None
        self.score_table = dict()

        self._beta_size = 0
        for index, row in enumerate(self.dataset):
            # TODO:
            public = [i for i in row if i not in private_item_list]
            private = [i for i in row if i in private_item_list]
            self.transactions.append(Transaction(index, public, private))

    def Sup(self, beta) -> int:
        """
        Returns the number of transactions in beta-cohort

        :param beta: Combination(subset) of public items no more than "p"
        :return: Sup(beta), a.k.a. k
        """
        k = 0
        # t means transaction, which is any single row of the dataset
        for t in self.transactions:
            if set(t.public).issuperset(beta):
                k += 1
        return k

    def p_breach(self, beta, private_item, k) -> float:
        """
        P(β→e)=Sup(β∪{e})/Sup(β)
        The probability that a transaction contains e, given that it contains β

        :param beta: Combination of public items no more than "p"
        :param private_item: A single private item
        :param k: Sup(beta) a.k.a number of transactions in beta-cohort
        :return: Breach probability of beta
        """
        temp_beta = beta.copy()
        temp_beta.append(private_item)
        return self.Sup(temp_beta) / k

    def is_mole(self, beta) -> bool:
        """

        :param beta: Combination(subset) of public items no more than "p"
        :return:
        """

        k = self.Sup(beta)

        if k == 0:
            return False
        # TODO: decide if using < or <= makes more sense
        if k < self.k:
            return True
        for e in self.private_item_list:
            if self.p_breach(beta, e, k) > self.h:
                return True
        return False

    def suppress_size1_moles(self):
        """
        # suppress all the public items that are size-1 moles
        :return: list of size1 moles (just in case I have use for them down the line)
        """
        # size1_moles = list()
        start_time = time.time()
        print("Started suppressing size-1 moles")
        for e in self.public_item_list:

            if self.is_mole([e]):
                # delete item e from all transactions
                self.size1_moles.append(e)
                for t in self.transactions:
                    if e in t.public:
                        t.public.remove(e)
                    # try:
                    #     t.public.remove(e)
                    # except ValueError:
                    #     continue
        print(
            f"Suppressed {len(self.size1_moles)} size-1 mole public items. Time passed: {int(time.time() - start_time)}")

    # MM(e) is the number of minimal moles containing the item e
    def find_minimal_moles(self):

        start_time = time.time()
        print("Started identification process for Minimal moles and Extendible non-moles")
        # C1 initial candidates
        # M1 init minimal moles
        # F1 init extendible non-moles

        C1 = [i for i in self.public_item_list if i not in self.size1_moles]
        # C1 = [1,2,3,4,5,6,7,8,9,10]
        M1 = list()
        F1 = list()

        for e in C1:
            if self.is_mole([e]):
                M1.append([e])
            else:
                F1.append([e])

        F = [F1]
        M = [M1]
        del F1, M1, C1
        # FIXME: this may need to start from 1 instead of 0
        i = 0
        while i < self.p and len(F[i]) > 0:
            print(f"\nStarted i:{i} and len(F[i]):{len(F[i])}")
            time_a = time.time()
            # generate Canditate set for Mi+1 and Fi+1
            F_, M_ = self.foo(F[i], M[i])
            F.append(F_)
            M.append(M_)
            i += 1
            print(f"M-F calculation time for i:{i} is {time.time() - time_a} seconds")

        print(f"Finished find minimal moles in {time.time() - start_time} seconds")

        # no need to return F
        print(f"FINAL M:{M}\n")
        counter = 0
        for i in M:
            print(f"len(M[{counter}]):{len(i)}")
            counter += 1



        self.moles = M
        return M

    def foo(self, F, M):
        """Return (Fi+1, Mi+1)
        """
        M1 = list()
        F1 = list()
        C1 = self.generate_C(F, M)
        for beta in C1:
            if self.is_mole(beta):
                M1.append(beta)
            else:
                F1.append(beta)
        return F1, M1

    @staticmethod
    def find_subsets_of_size_n(l: list, n: int):
        return itertools.combinations(l, n)

    @staticmethod
    def diff_list(L1, L2):
        return len(set(L1).symmetric_difference(set(L2)))

    # @staticmethod
    def generate_C(self, F: list, M: list) -> list:
        """

        :param F: List of extendible moles
        :param M: List of minimal moles
        :return: Candidate list Ci+1
        """
        # this is basically a list of betas
        # for Fi and Fi+1

        # print(len(F))
        # print(len(F[2]))
        start_time = time.time()
        print(f"Started generating Candidate list, len(M):{len(M)}, len(F):{len(F)}")
        C = list()
        # TODO: Use find subsets function to calculate the Fi+1
        # l = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 6]]
        F_items = set()
        for i in F:
            for j in i:
                F_items.add(j)

        # Length of Fi+1 beta
        n = len(F[0]) + 1
        print(n)

        size_n_subsets = find_subarrays_of_size_n(list(F_items),n)
        for subset in size_n_subsets:
            flag = False
            for m in M:
                if set(subset).issuperset(m):
                    flag = True
                    continue
            if not flag:
                C.append(list(subset))

        # for i in range(len(F)):

            # for j in range(i + 1, len(F)):
            #     print(f"Fi:{F[i]}, Fi+1:{F[j]}")
            #     if self.diff_list(F[i], F[j]) == 2:
            #         new_F = list(F[i])
            #         new_F.extend(x for x in F[j] if x not in new_F)
            #         flag = False
            #         for m in M:
            #             # check if the Fi+1 is a superset of any mole in M
            #             if set(new_F).issuperset(m):
            #                 flag = True
            #                 break
            #         if not flag:
            #             C.append(new_F)
        # self.find_subsets_of_size_n(F[self._beta_size],se)
        print(f"Finished generating candidate list len(C):{len(C)}, time-passed:{time.time() - start_time}")
        return C

    def info_loss(self, e):
        """
        A function for calculating the information loss upon suppressing a public item e
        :param e: Public item e
        :return:
        """
        # IL(e) = Sup(e)
        return self.Sup(e)

    def build_mole_tree(self):
        pass

    def pipeline(self):
        # suppress minimal moles
        self.suppress_size1_moles()

        # find minimal moles M* from D
        min_moles = self.find_minimal_moles()

        # TODO: build mole-tree for min moles
        return None


def build_tree(M):
    score_table = {}
    root = Node()
    for i in M:
        print(i)


    # pass


if __name__ == "__main__":
    DATA_PATH = "./Dataset/T40I10D100K_100.txt"

    # find unique items to randomly choose the private items among them
    unique_items = find_unique_items(DATA_PATH)
    unique_items = [int(i) for i in unique_items]
    print(f"Number of Unique items: {len(unique_items)}")

    # randomly choose private items
    # dataset =
    np.random.seed(42)
    private_items = np.random.choice(unique_items, replace=False, size=10)
    public_items = [i for i in unique_items if i not in private_items]
    public_items = public_items[:50]
    print(f"private: {len(private_items)}\npublic: {len(public_items)}")

    dataset = []
    with open(DATA_PATH, "r") as file:
        for line in file:
            dataset.append([int(i) for i in set(line.rstrip().split())])

    hkp = HKPCoherence(dataset, public_items, private_items, h=0.8, k=2, p=3)

    hkp.suppress_size1_moles()
    M_star = hkp.find_minimal_moles()

    with open("hkp_pickle.pkl", "wb") as f:
        pickle.dump(hkp, f)



    # l = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 6]]
    # root = Node(label="root")
    # # print(m1.children ==)
    # if not root.children:
    #     print("empty")
    # print(l[0][1])
    # build_tree(l)

