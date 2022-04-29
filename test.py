from functools import reduce

import numpy as np
import time
import itertools
import pickle

from anytree import AnyNode, NodeMixin, RenderTree, search

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
    # TODO: should I save the suppressed items into a global variable?
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
        self.MM = dict()
        self.score_table = None
        self.mole_tree_root = None
        self.suppressed_items = None

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
        # return M

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
    def all_paths(start_node: Node) -> list:
        """
        Finds all the paths from the starting node to the leaf nodes

        :param start_node:
        :return: A list of paths
        """
        skip = len(start_node.path) - 1
        return [leaf.path[skip:] for leaf in search.PreOrderIter(start_node, filter_=lambda node: node.is_leaf)]

    @staticmethod
    def find_subsets_of_size_n(l: list, n: int):
        """
        Generates subsets of size-n given a list

        :param l: List of items to make subsets of
        :param n: Size of a subset
        :return:
        """
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

        size_n_subsets = find_subarrays_of_size_n(list(F_items), n)
        for subset in size_n_subsets:
            flag = False
            for m in M:
                if set(subset).issuperset(m):
                    flag = True
                    continue
            if not flag:
                C.append(list(subset))

        print(f"Finished generating candidate list len(C):{len(C)}, time-passed:{time.time() - start_time}")
        return C

    def MM_e(self, e: int, min_moles: list) -> int:
        """
        Returns the number of minimal moles containing the public item e

        :param e: Public item
        :param min_moles: List of minimal moles
        :return:
        """
        count = 0
        # i is the mole length
        for i in min_moles:
            for mole in i:
                if e in mole:
                    count += 1

        return count

    def MM_desc_order(self, min_moles: list) -> dict:
        """
        Function that returns public items and their respected MM(e) counts in descending order

        :return:
        """

        # items = set()
        items_mm_count = dict()
        ordered_moles = dict()
        # find the public items that are parts of minimal moles
        # i denotes the size of the mole (beta)
        for mole_level, i in enumerate(min_moles):
            # ordered_moles[mole_level] = dict()
            for mole in i:
                for item in mole:
                    if item not in items_mm_count:
                        items_mm_count[item] = self.MM_e(item, min_moles)

        # sort moles in descending order of cumulative MM
        for mole_level, i in enumerate(min_moles):
            mole_dict = dict()
            for index, mole in enumerate(i):
                mole_mm_count = 0
                mole_dict[index] = dict()
                for item in mole:
                    mole_mm_count += items_mm_count.get(item)
                mole_dict[index]["mm_count"] = mole_mm_count
                mole_dict[index]["mole"] = sorted(mole, key=lambda x: items_mm_count.get(x), reverse=True)
            ordered_moles[mole_level] = dict(
                sorted(mole_dict.items(), key=lambda itm: itm[1]["mm_count"], reverse=True))

        self.MM = items_mm_count
        return ordered_moles

    def info_loss(self, e):
        """
        A function for calculating the information loss for suppressing a public item e

        :param e: Public item e
        :return:
        """
        # IL(e) = Sup(e)
        return self.Sup(e)

    def calculate_mole_num(self, node: Node) -> int:
        """
        Calculates the mole_num attribute of a given mole tree node

        :param node: A node of the mole tree
        :return:
        """
        return len(self.all_paths(node))

    @staticmethod
    def get_last_node_link(label, score_table: dict) -> Node:
        """
        Finds the last node of the nodelink given a public item

        :param label:
        :param score_table:
        :return:
        """

        head_node = score_table[label]["head_of_link"]

        current_node = head_node
        next_node = current_node.node_link

        while next_node is not None:
            current_node = next_node
            next_node = current_node.node_link

        return current_node

    # TODO: delete this func, it won't be used, most likely
    @staticmethod
    def find_root_connected_link_node(label, score_table: dict):
        """

        :param label:
        :param score_table:
        :return:
        """
        head_node = score_table[label]["head_of_link"]

        current_node = head_node
        next_node = current_node.node_link

        while next_node is not None and next_node.parent.label != "root":
            current_node = next_node
            next_node = current_node.node_link

        if current_node.parent.is_root:
            return current_node
        else:
            return None

    @staticmethod
    def get_head_of_link(label, score_table: dict) -> Node:
        """
        Finds the head node of an item from the scoretable

        :param label:
        :param score_table:
        :return:
        """
        return score_table[label]["head_of_link"]

    def build_mole_tree(self):
        """

        :return:
        """
        score_table = dict()
        root = Node(label='root')
        parent_node = None
        # MM(e) rankings of the items that are parts of the moles
        M_star = self.MM_desc_order(self.moles)
        # print(M_sta)
        for mole_level in M_star.values():
            # print(f"mole level : {mole_level}")
            for mole in mole_level.values():
                # print(f"------\nmole: {mole['mole']}\nrest: {mole['mole'][1:]}")
                # make first item of the mole a parent for the rest of the items
                nodes = list()
                for index, item in enumerate(mole["mole"]):

                    if item not in score_table.keys():
                        score_table[item] = dict()
                        score_table[item]["MM"] = self.MM.get(item)
                        score_table[item]["IL"] = self.info_loss([item])

                        # make the first item of the mole a direct child of the root node
                        if index == 0:
                            # print(item)
                            node = Node(label=item,
                                        mole_num=0,
                                        node_link=None,
                                        parent=root)

                            # if the item is not  in the score table yet, register it to the table and
                            # make the node head of link
                            score_table[item]["head_of_link"] = node
                            nodes.append(node)
                        else:
                            # print(f"item:{item}, nodes[index - 1]: {nodes[index - 1].label}")
                            # every item that comes after the first one should be the child of the following item
                            node = Node(label=item,
                                        mole_num=0,
                                        node_link=None,
                                        parent=nodes[index - 1])
                            # if the item is not  in the score table yet, register it to the table and
                            # make the node head of link
                            score_table[item]["head_of_link"] = node
                            nodes.append(node)

                    else:

                        if index == 0:

                            if item in [child.label for child in root.children]:

                                node = [child for child in root.children if item == child.label][0]
                                # print(type(node))
                                assert isinstance(node, Node)
                                nodes.append(node)

                            else:

                                node = Node(label=item,
                                            mole_num=0,
                                            node_link=None,
                                            parent=root)
                                # add new node to the nodelink
                                last_node = self.get_last_node_link(item, score_table)
                                last_node.node_link = node

                                nodes.append(node)

                        else:

                            # check if the children of nodes[index - 1] has this item as a child already
                            if item in [child.label for child in nodes[index - 1].children]:
                                node = [child for child in nodes[index - 1].children if item == child.label][0]
                                assert isinstance(node, Node)
                                nodes.append(node)

                            else:
                                # if is item not in nodes[index-1].children then we need to create a new node
                                node = Node(label=item,
                                            mole_num=0,
                                            node_link=None,
                                            parent=nodes[index - 1])

                                # add new node to the nodelink
                                last_node = self.get_last_node_link(item, score_table)
                                last_node.node_link = node
                                nodes.append(node)

        # travel all nodes of the tree, and assign mole numbers to each node
        for node in search.PreOrderIter(root):
            node.mole_num = self.calculate_mole_num(node)
            # print(f"Node: {node.label}, Mole num: {node.mole_num}")

        # root suppose to have no fields; mole_num, nodelink etc
        root.mole_num = None

        # sort items in scoretable in decreasing order or MM/IL
        # print(score_table)
        score_table = dict(sorted(score_table.items(), key=lambda x: x[1]["MM"] / x[1]["IL"], reverse=True))
        # print(score_table)
        self.mole_tree_root = root
        self.score_table = score_table

        # print(RenderTree(root))
        # self.print_tree(root)
        # print(f"Score Table: {score_table}")

        # find rankings for the moles in the minimal moles list

    @staticmethod
    def print_tree(start_node: Node):
        for pre, _, node in RenderTree(start_node):
            treestr = u"%s%s" % (pre, node.label)
            print(treestr.ljust(8))

    @staticmethod
    def node_link_length(head_link):
        temp = head_link  # Initialise temp
        count = 0  # Initialise count

        # Loop while end of linked list is not reached
        while temp:
            count += 1
            temp = temp.node_link
        return count

    def delete_subtree(self, node: Node, score_table: dict):

        # node.parent = None
        node_iter = [i for i in search.PreOrderIter(node)]
        # node.parent = None
        # print(f"{'-' * 15} Delete Subtree  {'-' * 50}\n Starting Node: {node.label}, "
        #       f"MM: {score_table[node.label]['MM']} "
        #       f"Subtree-length: {len(node_iter)}\nSubtree: ")

        # save mole num of start node so we can use ite later on with ancestor mole nums
        start_node_mole_num = node.mole_num

        self.print_tree(node)
        print(f"{'-' * 25}")
        # self.print_tree(node)
        print(f"######### Subtree iteration #########")
        for w in node_iter:
            print("*************** subtree w")
            assert w.label in score_table, f"Subtree node: {w.label} with mole_num: {w.mole_num} was not in score-table"
            print(f"w: {w}\nw.label: {w.label}\nw.mole_num: {w.mole_num}")
            print("***************")

            score_table[w.label]["MM"] -= w.mole_num
            w.mole_num = 0
            print(f"-CHANGE- w label: {w.label}, w MM: {score_table[w.label]['MM']}")
            if score_table[w.label]["MM"] == 0:
                item = score_table.pop(w.label, None)
                leftover = search.findall(self.mole_tree_root, filter_=lambda x: x.label == w.label)
                leftover_total_mole_num = reduce(lambda x,y: x+y, [i.mole_num for i in leftover])
                print(f"SUBTREE ITEM: labeled:{w.label} -> {item} WAS POPPED FROM THE SCORE TABLE IN SUBTREE ITERATION")
                print(score_table.keys())
                print(f"SUBTREE LEFTOVERS FOR ITEM: {w.label}, count: {len(leftover)}, MM: 0, Leftover total Mole_num: {leftover_total_mole_num}")
                # assert not leftover

        ancestors = [ancestor for ancestor in node.ancestors if not ancestor.is_root]
        # current_node_mole_num = self.calculate_mole_num(node)
        print(f"######### Ancestor mole_num update #########")
        for w in ancestors:
            # def test_st(): return True if w.label in score_table else False
            print("*************** Ancestor w")
            assert w.label in score_table, f"Ancestor: {w.label} with mole_num: {w.mole_num} was not in score-table "
            print(f"w: {w}\nw.label: {w.label}\nw.mole_num: {w.mole_num} ")
            print("***************")
            # One of the ancestors was an item that was already deleted from the score table since mole-num == 0
            print(f"w: {w}\nw.label: {w.label}\nw.mole_num: {w.mole_num}")
            # FIXME node.mole_num becomes 0 after the first iteration so this does not wokr properly
            # print(f"Ancestor: {w.label}, Ancs mole-num: {w.mole_num}, Ancs MM: {score_table[w.label]['MM']}\n"
            #       f"Node: {node.label}, Node mole-num: {node.mole_num} Node MM: {score_table[node.label]['MM']}\n")
            score_table[w.label]["MM"] -= start_node_mole_num
            w.mole_num -= start_node_mole_num
            print(
                f"-CHANGE- Ancestor label: {w.label}, Ancs mole-num: {w.mole_num}, Ancs MM: {score_table[w.label]['MM']}")
            if w.mole_num == 0:
                w.parent = None
                print(f"Node w ancestor: {w.label}, {w}\n"
                      f"removed from the tree")
            if score_table[w.label]["MM"] == 0:
                print("item MM zero")
                print(f"w: {w}\nw.label: {w.label}\nw.mole_num: {w.mole_num}")
                # assert not search.findall(self.mole_tree_root, filter_=lambda x: x.label==)
                item = score_table.pop(w.label,None)
                print(f"ITEM: {w.label} -> {item} WAS POPPED FROM THE SCORE TABLE IN ANCESTOR UPDATE")
                leftover = search.findall(self.mole_tree_root, filter_=lambda x: x.label == w.label)
                print(
                    f"ANCESTOR ITEM: labeled:{w.label} -> {item} WAS POPPED FROM THE SCORE TABLE IN SUBTREE ITERATION")
                print(f"ANCESTOR LEFTOVERS FOR ITEM: {w.label}, count: {len(leftover)}, MM: 0")

    def execute_algorithm(self):
        # suppress minimal moles
        self.suppress_size1_moles()

        # find minimal moles M* from D
        self.find_minimal_moles()

        # Build the mole tree
        self.build_mole_tree()

        suppressed_items = set()

        score_table = self.score_table
        root = self.mole_tree_root

        while score_table:
            print(f"Score-table length: {len(score_table)}")
            key, value = next(iter(score_table.items()))

            # add the item e with the max MM/IL to suppressed items set
            # scoretable is already sorted in dec order of MM/IL
            suppressed_items.add(key)
            print(f"SUPPRESSED ITEMS LIST: {suppressed_items}")
            node_link_list = list()
            ancestor_list = list()

            # To delete a node from the tree we can set its parent to NONE, so the node and
            # all of its following branches will be disconnected from the rest of the tree

            # get the headlink of the key
            head_link = self.get_head_of_link(key, score_table)
            ancestor_list.append([ancestor for ancestor in head_link.ancestors if not ancestor.is_root])
            # print(f"head link label: {head_link.label}, node: {head_link}")
            test_len = self.node_link_length(head_link)
            # we need to delete all the subtrees starting from headlink, and following the nodelink
            print(
                f"ITEM: {head_link.label}, NODELINK LEN: {self.node_link_length(head_link)}############################################################")
            # node_link_list.append(head_link)
            current_node = head_link
            while current_node is not None:
                # ancestor_list.append([ancestor for ancestor in current_node.ancestors if not ancestor.is_root])
                # disconnect the node from its parent, this will cut all the branch of the tree starting from this node
                # current_node.parent = None

                node_link_list.append(current_node)
                # TODO: update mole_num of ancestors
                # get the ancestor nodes before passing None to parents
                # print(f"##############\n"
                #       f"Current node: {current_node.label} "
                #       f"mole_num: {current_node.mole_num}\n"
                #       f"'\naddress: {current_node}"
                #       f"###############")
                self.delete_subtree(current_node, score_table)
                print(f"Current node link len: {self.node_link_length(current_node)} after deletion of previous one: \n")
                current_node = current_node.node_link
                # break

            self.print_tree(root)
            score_table = dict(sorted(score_table.items(), key=lambda x: x[1]["MM"] / x[1]["IL"], reverse=True))

            # break

            # assert not search.findall(root, filter_=lambda node: node.label == head_link.label)

            # finally delete the public item e from the score table, so we can move to the next one
        print(score_table)


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

    hkp = HKPCoherence(dataset, public_items, private_items, h=0.8, k=2, p=2)

    # hkp.suppress_size1_moles()
    # M_star = hkp.find_minimal_moles()
    # hkp.build_mole_tree()
    hkp.execute_algorithm()

    # n0 = Node("n0")
    # n1 = Node("n1")
    # n2 = Node("n2")
    #
    # n0.node_link = n1
    # n1.node_link = n2
    # test_dict = {"n0": {"head_of_link": n0}}
    # print(f"last node == {hkp.get_last_node_link(n0.label, test_dict).node_link}")

    # with open("hkp_pickle.pkl", "wb") as f:
    #     pickle.dump(hkp, f)

    # l = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 6]]
    # root = Node(label="root")
    # # print(m1.children ==)
    # if not root.children:
    #     print("empty")
    # print(l[0][1])
    # build_tree(l)