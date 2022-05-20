from functools import reduce

import numpy as np
import time
import itertools
import pickle

from anytree import NodeMixin, RenderTree, search

import utils

'''
    #######- GREEDY ALGORITHM -#######
    A public item must be suppressed, if the item on ITS OWN IS A MOLE.
        If a public item is a (size-1) mole, the item will
        not occur in any (h,k,p)-cohesion of D, thus, can be suppressed in
        a preprocessing step


'''


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
        # self.original_mole_tree = None
        self.suppressed_items = None
        self.finished_public_items = None
        self.total_occurrence_count = 0  # sum of all item occurrences in dataset
        self.suppressed_item_occurrence_count = 0  # supp item occurrence count
        for index, row in enumerate(self.dataset):
            # TODO:
            public = [i for i in row if i not in private_item_list]
            private = [i for i in row if i in private_item_list]
            self.transactions.append(Transaction(index, public, private))
            self.total_occurrence_count += len(row)

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

    def anonymization_verifier(self):

        C1 = self.finished_public_items
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
            F_, M_ = self.generate_M_F(F[i], M[i])
            F.append(F_)
            M.append(M_)
            i += 1
            print(f"M-F calculation time for i:{i} is {time.time() - time_a} seconds")

        anonymized = True
        for possible_mole in M:
            if len(possible_mole) != 0:
                print(f"Anonymization FAILED, size - {len(possible_mole)} moles: {possible_mole}")
                anonymized = False
        if anonymized:
            print(f"Anonymization SUCCEED, parameters - h: {self.h}, k: {self.k}, p: {self.p}")

        # pass

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
            print(f"\nStarted i:{i}, len(F[{i}]):{len(F[i])}, len(M[{i}]):{len(M[i])}")
            time_a = time.time()
            # generate Canditate set for Mi+1 and Fi+1
            F_, M_ = self.generate_M_F(F[i], M[i])
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

    def generate_M_F(self, F, M):
        """Return (Fi+1, Mi+1)
        """
        print(f"Generate M-F, F: {len(F)}, M: {len(M)}")
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
        temp = [leaf.path[skip:] for leaf in search.PreOrderIter(start_node, filter_=lambda node: node.is_leaf)]
        print(f"label: {start_node.label}, mole_num: {len(temp)}")
        return temp

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
        # unique_items = set()
        # for i in range(len(F)):
        #     for j in F[i]:
        #         unique_items.add(j)
        #
        # C = list()
        # for i in range(len(F)):
        #     for j in unique_items:
        #         new_F = list(F[i])
        #         if j not in new_F:
        #             new_F.append(j)
        #             flag = False
        #
        #             for m in M:
        #                 if set(new_F).issuperset(m):
        #                     flag = True
        #                     break
        #             if not flag:
        #                 C.append(new_F)

        # C = list()
        # F_items = set()
        # for i in F:
        #     for j in i:
        #         F_items.add(j)
        #
        # # Length of Fi+1 beta
        # n = len(F[0]) + 1
        # print(n)
        #
        # size_n_subsets = utils.find_subarrays_of_size_n(list(F_items), n)
        # for subset in size_n_subsets:
        #     flag = False
        #     for m in M:
        #         if set(subset).issuperset(m):
        #             flag = True
        #             continue
        #     if not flag:
        #         C.append(list(subset))


        C = list()
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                if self.diff_list(F[i], F[j]) == 2:
                    new_F = list(F[i])
                    new_F.extend(x for x in F[j] if x not in new_F)
                    flag = False
                    for m in M:
                        if set(new_F).issuperset(m):
                            flag = True
                            break
                    if not flag:
                        C.append(new_F)

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

    def calculate_distortion(self) -> float:
        # S/N
        # S sum info loss of suppressed items
        # N occurrence of all items
        return self.suppressed_item_occurrence_count / self.total_occurrence_count


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

        with open("Pickles/hkp_MM.pkl", "wb") as f:
            pickle.dump(self.MM, f)

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
delete
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
        # MM(e) rankings of the items that are parts of the moles
        M_star = self.MM_desc_order(self.moles)

        with open("Pickles/hkp_minimal_moles.pkl", "wb") as f:
            pickle.dump(M_star, f)

        # print(M_sta)
        for mole_level in M_star.values():
            # print(f"mole level : {mole_level}")
            for mole in mole_level.values():
                # print(f"------\nmole: {mole['mole']}\nrest: {mole['mole'][1:]}")
                # make first item of the mole a parent for the rest of the items
                nodes = list()
                for index, item in enumerate(mole["mole"]):

                    # if item in mole is yet to be registered to the score table
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
        # self.original_mole_tree = root.copy()
        self.score_table = score_table

        # FIXME: While pickling max recursion error is raised
        # with open("Pickles/hkp_mole_tree_root.pkl", "wb") as f:
        #     pickle.dump(self.mole_tree_root, f)

        # with open("Pickles/hkp_score_table.pkl", "wb") as f:
        #     pickle.dump(self.score_table, f)

        # print(RenderTree(root))
        # self.print_tree(root)
        # print(f"Score Table: {score_table}")

        # find rankings for the moles in the minimal moles list

    @staticmethod
    def print_tree(start_node: Node):
        print(f"\nTree Starting from Node: {start_node.label}")
        for pre, _, node in RenderTree(start_node):
            treestr = u"%s%s" % (pre, node.label)
            print(treestr.ljust(8))

    @staticmethod
    def node_link_length(head_link):
        temp = head_link  # Initialise temp
        count = 0  # Initialise count

        if not head_link:
            return count

        else:
            # Loop while end of linked list is not reached
            while temp:
                count += 1
                temp = temp.node_link
            return count

    def delete_subtree(self, node: Node, score_table: dict):
        # TODO: Check if the node.parent is NONE, if none, it means that
        print("\n-------- Initiate delete_subtree -------")
        # assert isinstance(node.parent,
        #                   Node), f"Current Node label -> {node.label}, Ancestors : {[ancestor.label for ancestor in node.ancestors if not ancestor.is_root]}"
        # print(f"Delete subtree of item: {node.label}, mole_num: {node.mole_num}, "
        #       f"parent: {node.parent.label}, node_link: {node.node_link}")

        # nodes in the subtree at the node
        node_iter = [i for i in search.PreOrderIter(node)]

        # all ancestors of node
        ancestors = [ancestor for ancestor in node.ancestors if not ancestor.is_root]

        # save mole_num of start node, so we can use ite later on with ancestor mole nums
        start_node_mole_num = node.mole_num

        # cut the subtree at node from the complete mole tree
        # setting the parent of the node to None is an easy way of removing the subtree
        # node.parent = None

        self.print_tree(node)
        print(f"{'-' * 25}")
        # self.print_tree(node)
        print(f"\n######### Subtree iteration #########")
        for w in node_iter:
            # Check if the item is still in the scoretable
            if w.label not in score_table:
                print(f"SUBTREE ITEM, label: {w.label}, mole_num: {w.mole_num} NOT IN THE SCORE TABLE")
                continue

            else:
                print(f"SUBTREE ITEM, label: {w.label}, mole_num: {w.mole_num}, MM: {score_table[w.label]['MM']}")
                # if the item is in scoretable reduce MM for the item by mole_num
                score_table[w.label]["MM"] -= w.mole_num

                # if MM(e) hits zero on scoretable, then remove the item from the scoretable
                # If mole_num hits zero or lower cut the subtree from the tree

                if score_table[w.label]["MM"] <= 0:
                    item = score_table.pop(w.label, None)
                    print(
                        f"SUBTREE ITEM: labeled: {w.label} -> {item} WAS POPPED FROM THE SCORE TABLE IN SUBTREE ITERATION")
                    print(
                        f"Updated Score-table: {score_table.keys()}\nUpdated Score-table length: {len(score_table.keys())}")

            # w.mole_num -= w.mole_num
            # print(f"-CHANGE- w label: {w.label}, w MM: {score_table[w.label]['MM']}")

        # current_node_mole_num = self.calculate_mole_num(node)
        print(f"\n######### Ancestor mole_num update #########")
        for w in ancestors:

            if w.label not in score_table:
                print(f"ANCESTOR ITEM, label: {w.label}, mole_num: {w.mole_num} NOT IN THE SCORE TABLE")
                # If item w not in scoretable, cut the node
                continue
            else:
                print(f"ANCESTOR ITEM, label: {w.label}, mole_num: {w.mole_num}, MM: {score_table[w.label]['MM']}, "
                      f"LINK NODE: {node.label}, LINK NODE MOLE_NUM: {start_node_mole_num}")
                score_table[w.label]["MM"] -= start_node_mole_num
                w.mole_num -= start_node_mole_num
                # if w.mole_num < 0:
                #     print(f"###--- ANCESTOR ITEM {w.label}, mole_num below ZERO ---###")
                # elif score_table[w.label]["MM"] < 0:
                #     print(
                #         f"###--- ANCESTOR ITEM {w.label}, MM: {score_table[w.label]['MM']}, MM below ZERO ---###")

                if w.mole_num == 0:
                    w.parent = None
                    print(f"Ancestor node w : {w.label}, mole_num: {w.mole_num} {w}"
                          f"was cut from the tree")
                if score_table[w.label]["MM"] <= 0:
                    item = score_table.pop(w.label, None)
                    print(f"ANCESTOR ITEM: {w.label} -> {item} WAS POPPED FROM THE SCORE TABLE IN ANCESTOR UPDATE")
                    print(f"Updated Score-table length: {score_table.keys()}")
        print("----------------------------")

    def execute_algorithm(self):
        time_dict = dict()
        # suppress minimal moles
        self.suppress_size1_moles()

        start_time = time.time()
        # find minimal moles M* from D
        self.find_minimal_moles()
        time_passed = int(start_time - time.time())
        time_dict["find_min_moles"] = time_passed

        start_time = time.time()
        # Build the mole tree
        self.build_mole_tree()
        time_passed = int(start_time - time.time())
        time_dict["build_mole_tree"] = time_passed

        suppressed_items = set()

        score_table = self.score_table
        root = self.mole_tree_root

        self.print_tree(root)
        while score_table:
            # Get the next item in the score_table. Since we
            key, value = next(iter(score_table.items()))
            print(f"\n----------- INITIATE SUPPRESSION OF ITEM {key} FROM SCORE TABLE -----------")
            print(f"Score-table length is {len(score_table)} before suppression of item {key}")

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
                f"ITEM: {head_link.label}, NODE LINK LEN: {self.node_link_length(head_link)}")
            # node_link_list.append(head_link)
            current_node = head_link
            while current_node is not None:
                node_link_list.append(current_node)

                self.delete_subtree(current_node, score_table)
                print(f"UPDATED NODE LINK LEN: {self.node_link_length(current_node)} "
                      f"FOR ITEM: {current_node.label}\n")
                current_node = current_node.node_link

                # break

            self.print_tree(root)
            score_table = dict(sorted(score_table.items(), key=lambda x: x[1]["MM"] / x[1]["IL"], reverse=True))
            print(f"Items in Score-table: {score_table.keys()}\n"
                  f"Score-table length: {len(score_table.keys())}")

            # break

            # assert not search.findall(root, filter_=lambda node: node.label == head_link.label)

            # finally delete the public item e from the score table, so we can move to the next one
        if not score_table:
            print(f"\nScore table is empty.\nSuppressed items: {suppressed_items}\n"
                  f"Suppressed items length: {len(suppressed_items)}")
            self.suppressed_items = suppressed_items
            self.finished_public_items = [i for i in self.public_item_list if i not in suppressed_items]
            for row in self.dataset:
                for item in row:
                    if item in self.suppressed_items:
                        self.suppressed_item_occurrence_count += 1
            self.print_tree(root)
            for node in search.PreOrderIter(root):
                print(f"Label: {node.label}, mole_num: {node.mole_num}")

            # Suppress all items in suppressed_items from the database D
            for e in suppressed_items:
                for t in self.transactions:
                    if e in t.public:
                        t.public.remove(e)

            distortion = self.calculate_distortion()
            print(f"DISTORTION: {distortion}")
        print("Preparing Anonymized txt file")
        with open(r'Dataset/Anonymized/public.txt', 'w') as f:
            for t in self.transactions:
                # merge public and private lists to get full transaction
                # merge = t.public + t.private
                merge = ' '.join(map(str, t.public))
                f.write(f"{merge}\n")
            print('Done')

        with open(r'Dataset/Anonymized/private.txt', 'w') as f:
            for t in self.transactions:
                # merge public and private lists to get full transaction
                # merge = t.public + t.private
                merge = ' '.join(map(str, t.private))
                f.write(f"{merge}\n")
            print('Done')
