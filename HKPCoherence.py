import time
import os
import pandas as pd
import utils

from anytree import NodeMixin, RenderTree, search, PreOrderIter
from itertools import chain, combinations
from collections import defaultdict

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


# # this can be an inner class of hkp coherence
# class Transaction:
#     # tag the private and pub items of each transaction in data
#     def __init__(self, ID: int, public: list, private: list):
#         self.ID = ID
#         self.public = public
#         self.private = private


class HKPCoherence:
    def __init__(self, dataset: list, public_item_list: list, private_item_list: list, h: float, k: int, p: int,
                 sigma: float):
        """

        :param dataset: The list of transactions
        :param public_item_list: Public items of the dataset
        :param private_item_list: Private items of the dataset
        :param h: The percentage of the transactions in beta-cohort that contain a common private item
        :param k: The least number of transactions that should be contained in the beta-cohort
        :param p: The maximum number of public items that can be obtained as prior knowledge in a single attack
        :param sigma: the percentage of public items wrt all items in the dataset
        """
        # includes all private and public items
        self.dataset = dataset
        self.public_item_list = public_item_list  # list of public items among all data
        self.private_item_list = private_item_list  # list of priv items among all data
        self.h = h
        self.k = k
        self.p = p
        self.sigma = sigma
        self.transactions = list()
        self.size1_moles = list()
        self.support_dict = None
        self.moles = None
        self.MM = dict()
        self.score_table = None
        self.mole_tree_root = None
        self.suppressed_items = None
        self.processed_public_items = None  # public items after suppression
        self.total_occurrence_count = 0  # sum of all item occurrences in dataset
        self.suppressed_item_occurrence_count = 0  # supp item occurrence count
        for index, row in enumerate(self.dataset):
            self.total_occurrence_count += len(row)

    def get_itemset_transaction_list(self, data_iterator, public_item_list, private_item_list):
        """
        Return frozenset versions of the data for faster process.
        For public and private item lists, each individual item is a frozenset

        :param data_iterator: Dataset
        :param public_item_list: The list of the public items
        :param private_item_list: The list of the private items
        :return: public_item_set, private_item_set, transaction_list
        """
        transaction_list = list()
        public_item_set = set()
        private_item_set = set()

        for record in data_iterator:
            transaction = frozenset(record)
            transaction_list.append(transaction)

        for item in public_item_list:
            public_item_set.add(frozenset([item]))

        for item in private_item_list:
            private_item_set.add(frozenset([item]))

        return public_item_set, private_item_set, transaction_list

    def suppress_size1_moles(self, public_item_set, private_item_set, transaction_list, support_set):
        """
        Calculate support and P_breach values for original size-1 public items as betas.
        Remove all the size-1 moles from all transactions in the dataset
        :returns: candidate set for finding min moles, and a transaction list without size-1 moles

        :param private_item_set:
        :param public_item_set:
        :param transaction_list:
        :param support_set: Support dictionary of all betas
        :return: _item_set, clean_transaction_list
        """
        # public?
        """
        how do we count sup for beta and beta->e in one go
        """

        _item_set, size1_moles = self.get_moles_and_candidates(public_item_set,
                                                               private_item_set,
                                                               transaction_list,
                                                               support_set,
                                                               )

        self.size1_moles = size1_moles
        print(f"Non-mole size-1 items len: {len(_item_set)}, items:{_item_set}")

        clean_transaction_list = []
        for transaction in transaction_list:
            temp_transaction = list(transaction)
            for item in size1_moles:
                if item.issubset(transaction):
                    temp_item = list(item)
                    temp_transaction.remove(temp_item[0])

            clean_transaction_list.append(frozenset(temp_transaction))

        print(f"len clean transaction list: {len(clean_transaction_list)}")

        return _item_set, clean_transaction_list

    @staticmethod
    def join_set(item_set, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set(
            [i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length]
        )

    def get_moles_and_candidates(self, item_set, private_item_set, transaction_list, support_set, mole_list=None):
        """
        Function to find F and M; extendible moles and minimal moles

        :param item_set: Set of extendible moles
        :param private_item_set:
        :param transaction_list:
        :param support_set:
        :param mole_list:
        :return: Fi+1, Mi+1
        """

        if mole_list is None:
            mole_list = set([])

        _item_set = set()
        local_set = defaultdict(int)
        minimal_moles = set()
        p_breach_dict = {}

        # FIXME: Operation below takes so much memory, when calculating for p==3 11 gigs of ram was full
        # create sets of beta->e for counting support for breach probability of beta
        for beta in item_set:
            p_breach_dict[beta] = defaultdict(int)
        #     for e in private_item_set:
        #         beta_e = beta.copy()
        #         beta_e = beta_e.union(e)
        #         p_breach_dict[beta][beta_e] = 0

        # calculate support for beta and beta->e
        for transaction in transaction_list:
            for beta in item_set:
                if beta.issubset(transaction):
                    support_set[beta] += 1
                    local_set[beta] += 1
                    for e in private_item_set:
                        if e.issubset(transaction):
                            p_breach_dict[beta][e] += 1

        # check if beta is a mole or not
        # if Sup(beta) < k or sup(beta->e)/sup(beta) > h, then mole
        for item, support in local_set.items():
            support_beta_e = max(p_breach_dict[item].values(), default=0)
            print(f"support beta->e: {support_beta_e}")

            if support < self.k or support_beta_e / support > self.h:
                flag = False
                # TODO: we may need to check for moles of all levels, got one MM != mole_num error even after checking for the previous level of moles
                for m in mole_list:
                    if m.issubset(item):
                        flag = True
                        break
                if not flag:
                    minimal_moles.add(item)
            else:
                # FIXME: this check may not be necessary, if if there is a problem with build_mole_tree first
                _item_set.add(item)

        print(f"Non-mole betas len: {len(_item_set)}, betas:{_item_set}")

        return _item_set, minimal_moles

    def find_minimal_moles(self):
        """
        Apriori algorithm like fast solution to find all the minimal moles in the dataset
        """

        # get the frozen set versions of dataset and the public items
        public_item_set, private_item_set, transaction_list = self.get_itemset_transaction_list(self.dataset,
                                                                                                self.public_item_list,
                                                                                                self.private_item_list)

        support_set = defaultdict(int)
        F = dict()
        M = dict()

        # public items that non-moles and transaction list cleaned from size-1 moles
        one_C_set, transaction_list = self.suppress_size1_moles(public_item_set,
                                                                private_item_set,
                                                                transaction_list,
                                                                support_set)

        start_time = time.time()
        current_F_set = one_C_set
        current_M_set = None
        p = 2

        while p <= self.p and current_F_set != set([]):
            F[p - 1] = current_F_set
            current_F_set = self.join_set(current_F_set, p)
            current_C_set, current_M_set = self.get_moles_and_candidates(
                current_F_set, private_item_set, transaction_list, support_set, current_M_set
            )
            current_F_set = current_C_set
            M[p] = current_M_set
            print(f"P: {p}, M length: {len(M[p])}")
            p += 1

        minimal_moles = []
        for p in M.keys():
            mole_level_container = []
            for mole in M[p]:
                mole_level_container.append(list(mole))
            minimal_moles.append(mole_level_container)

        pass_time = time.time() - start_time
        self.moles = minimal_moles

        print(f"Min-moles: {minimal_moles}")
        for mole_level in minimal_moles:
            print(mole_level)

        # TODO: remove items with len(beta) > 1 from support dict we dont need them
        self.support_dict = support_set
        return pass_time

    # TODO: Make a new anonymization verifier

    @staticmethod
    def all_paths(start_node: Node) -> list:
        """
        Finds all the paths from the starting node to the leaf nodes

        :param start_node:
        :return: A list of paths
        """
        skip = len(start_node.path) - 1
        temp = [leaf.path[skip:] for leaf in search.PreOrderIter(start_node, filter_=lambda node: node.is_leaf)]
        # print(f"label: {start_node.label}, mole_num: {len(temp)}")
        return temp

    def MM_e(self, e, min_moles: list) -> int:
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
        """
        Calculate the deviation of data from its true form after finishing the anonymization process

        :return:
        """
        # S/N
        # S sum info loss of suppressed items
        # N occurrence of all items
        return self.suppressed_item_occurrence_count / self.total_occurrence_count

    def MM_desc_order(self, min_moles: list) -> dict:
        """
        Function that returns public items and their respected MM(e) counts in descending order

        :return:
        """

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

        # sort moles in descending order of total MM
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

        # store MM value for each public item e
        self.MM = items_mm_count

        print(f"\nOrdered moles: {ordered_moles}\n")
        return ordered_moles

    def info_loss(self, e):
        """
        A function for calculating the information loss for suppressing a public item e

        :param e: Public item e
        :return:
        """
        # IL(e) = Sup(e)

        return self.support_dict[frozenset(e)]

    def calculate_mole_num(self, node: Node) -> int:
        """
        Calculates the mole_num attribute of a given mole tree node

        :param node: A node of the mole tree
        :return:
        """
        return len(self.all_paths(node))

    def delete_node_link(self, link_node, score_table: dict):
        """

        :param link_node: The node that will be removed from its node_link
        :param score_table:
        :return:
        """

        # store head node of node-link
        temp = self.get_head_of_link(link_node.label, score_table)

        # if head node itself is the node to be deleted
        # change the head node in score table
        # TODO: if the head_link is the only node left, head_link.node_link might be None, what to do if that is the case
        if temp is not None:
            if temp == link_node:
                # head_link.node_link points to the next in line node after the heads
                score_table[link_node.label]["head_of_link"] = temp.node_link
                return

        # search the link node to be deleted, keep track of the previous node
        # because we need to change the node_link of the node that comes #
        # previous to the node we are looking for
        # TODO: temp.node_link might be None, take take of it if that is the case
        prev_node = None
        while temp is not link_node:
            prev_node = temp
            temp = temp.node_link

        # connect node link of prev node to the node comes after the link_node
        # we are looking for
        # this way we will unlink the link_node from the node_link
        prev_node.node_link = temp.node_link

        temp = None

    def get_last_node_link(self, label, score_table: dict) -> Node:
        """
        Finds the last node of the node-link given a public item

        :param label: The label of the node we need the node-link of
        :param score_table:
        :return: The last node of the node-link
        """

        # get the head node of the node-link
        head_node = self.get_head_of_link(label, score_table)

        current_node = head_node
        next_node = current_node.node_link

        # traverse node-link to find the last item
        while next_node is not None:
            current_node = next_node
            next_node = current_node.node_link

        return current_node

    @staticmethod
    def get_head_of_link(label, score_table: dict) -> Node:
        """
        Gets the head node of an item from the scoretable

        :param label: The label of the node we need the head node of
        :param score_table:
        :return:
        """
        return score_table[label]["head_of_link"]

    def build_mole_tree(self):
        """
        Builds mole tree from the list of minimal moles identified

        :return:
        """

        # initiate score table
        score_table = dict()

        # define root node
        root = Node(label='root')

        # minimal moles arranged in the descending order of their MM values
        M_star = self.MM_desc_order(self.moles)

        # mole_level is the mole plane that divides the moles depending on their length; len 2 moles, len 3 moles ...
        for mole_level in M_star.values():
            # moles in the mole level
            for mole in mole_level.values():
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
        score_table = dict(sorted(score_table.items(), key=lambda x: x[1]["MM"] / x[1]["IL"], reverse=True))
        self.mole_tree_root = root
        self.score_table = score_table

    @staticmethod
    def print_tree(start_node: Node):
        """
        Print tree sprout from given node

        :param start_node:
        :return:
        """
        print(f"\nTree Starting from Node: {start_node.label}")
        for pre, _, node in RenderTree(start_node):
            treestr = u"%s%s:%s" % (pre, node.label, node.mole_num)
            print(treestr.ljust(8))

    @staticmethod
    def node_link_length(head_link):
        """
        Finds the length of a node_link

        :param head_link: Head node of the node_link
        :return:
        """
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
        """
        Function to delete subtree

        :param node:
        :param score_table:
        :return:
        """
        assert node.label in score_table.keys(), f"node: {node.label} was not in the score-table"

        print(f"\n---- Delete Subtree ----\n"
              f"node: {node.label}, mole_num: {node.mole_num}, node_link_len: {self.node_link_length(node)}, "
              f"MM: {score_table[node.label]['MM']}, "
              f"head_link: {'TRUE' if score_table[node.label]['head_of_link'] == node else 'FALSE'}")

        # delete all the minimal moles at the subtree node
        print("----# Subtree iter #----")
        for w in PreOrderIter(node):

            print(f"label: {w.label}, mole_num: {w.mole_num}, MM: {score_table[w.label]['MM']}")

            # decrease w.MM by w.mole_num
            score_table[w.label]["MM"] -= w.mole_num
            assert score_table[w.label]["MM"] >= 0, f"label: {w.label}, mole_num: {w.mole_num}, " \
                                                    f"MM: {score_table[w.label]['MM']}"

            # FIXME: the root of the errors might be heres
            # remove the processed nodes from their respective node_links
            # if w is not node:
            #     self.delete_node_link(link_node=w,
            #                           score_table=score_table)
            self.delete_node_link(link_node=w,
                                  score_table=score_table)
            # if MM == 0, then remove the item from the score-table
            if score_table[w.label]["MM"] == 0:
                print(f"label: {w.label}, node_link_len: {self.node_link_length(w)} MM has become 0, "
                      f"thus it will be removed from the score-table")
                del score_table[w.label]

        print("----# Ancestors of node iter #----")
        # find all the ancestors of the node
        ancestors = [ancestor for ancestor in node.ancestors]
        print(f"node.ancestors: {[node.label for node in ancestors]}")
        for w in ancestors:
            # Do not count the root node as an ancestor
            if not w.is_root:
                print(f"label: {w.label}, mole_num: {w.mole_num}")
                # decrement w.mole_num and w.MM by node.mole_num
                w.mole_num -= node.mole_num
                score_table[w.label]["MM"] -= node.mole_num

                assert score_table[w.label][
                           "MM"] >= 0 and w.mole_num >= 0, f"label: {w.label}, mole_num: {w.mole_num}, " \
                                                           f"MM: {score_table[w.label]['MM']}"

                # if mole_num hits 0, then cut node w from the tree
                if w.mole_num == 0:
                    print(f"label: {w.label}, mole_num has become 0, thus it will be removed from the tree")
                    w.parent = None

                    # if mole_num is 0 remove from the node link
                    # because it is useless
                    self.delete_node_link(link_node=w,
                                          score_table=score_table)
                # if MM of w hits 0 in score-table, remove w from the score-table
                if score_table[w.label]["MM"] == 0:
                    print(f"label: {w.label}, node_link_len: {self.node_link_length(w)} MM has become 0, "
                          f"thus it will be removed from the score-table")
                    del score_table[w.label]

        # lastly remove the node from tree
        node.parent = None
        print("----------------------------")

    def execute_algorithm(self):
        """
        Run complete anonymization algorithm
        :return:
        """
        performance_records = {"h": self.h,
                               "k": self.k,
                               "p": self.p,
                               "sigma": self.sigma,
                               "data_size": len(self.dataset),
                               "distortion": 0,
                               "time_find_min_moles": 0,
                               "time_total": 0}

        # find minimal moles M* from D
        pass_time = self.find_minimal_moles()
        performance_records["time_find_min_moles"] = int(pass_time)

        start_time = time.time()

        # Build the mole tree
        self.build_mole_tree()

        # initiate suppressed items set
        suppressed_items = set()

        score_table = self.score_table
        root = self.mole_tree_root

        # check if MM values in score table equal total mole_num count
        utils.check_MM_equal_mole_num(root, score_table)

        # process all items
        while score_table:

            # sort score table so item with max MM/IL is the next in line
            score_table = dict(sorted(score_table.items(), key=lambda x: x[1]["MM"] / x[1]["IL"], reverse=True))
            key, value = next(iter(score_table.items()))

            print(f"Score-table length is {len(score_table)} before suppression of item {key}")

            # add the item e with the max MM/IL to suppressed items set
            # scoretable is already sorted in dec order of MM/IL
            suppressed_items.add(key)
            print(f"SUPPRESSED ITEMS LIST: {suppressed_items}")

            # To delete a node from the tree we can set its parent to NONE, so the node and
            # all of its following branches will be disconnected from the rest of the tree

            # FIXME: Headlink was None, in one occasion, find out what is wrong
            # get the headlink of the key
            head_link = self.get_head_of_link(key, score_table)

            print(f"--------###### Node link details for item: {head_link.label} ######--------")
            temp = head_link
            while temp is not None:
                print(f"label: {temp.label}, mole_num: {temp.mole_num}, node_link: {temp.node_link}")
                temp = temp.node_link

            # delete all the subtrees starting from headlink, and following the nodelink
            current_node = head_link
            while current_node is not None:
                self.delete_subtree(current_node, score_table)
                current_node = current_node.node_link

            self.print_tree(root)
            print(f"Items in Score-table: {score_table.keys()}\n"
                  f"Score-table length: {len(score_table.keys())}")

        # after processing all items in the score-table, remove the items tagged as suppressed items from
        # the transactions in order to anonymize
        if not score_table:
            print(f"\nScore table is empty.\nSuppressed items: {suppressed_items}\n"
                  f"Suppressed items length: {len(suppressed_items)}, Size-1 moles: {len(self.size1_moles)}, "
                  f"Total Suppressed: {len(suppressed_items) + len(self.size1_moles)}")
            self.suppressed_items = suppressed_items
            self.processed_public_items = [i for i in self.public_item_list if
                                           i not in suppressed_items and i not in self.size1_moles]
            for row in self.dataset:
                for item in row:
                    if item in self.suppressed_items:
                        self.suppressed_item_occurrence_count += 1

            # Suppress all items in suppressed_items from the database D
            for e in suppressed_items:
                for t in self.dataset:
                    if e in t:
                        t.remove(e)

            distortion = self.calculate_distortion()
            performance_records["time_total"] = int(time.time() - start_time)
            performance_records["distortion"] = distortion

            # Write performance records to csv file for later use
            # Open the file in "append" mode
            output_path = "./Plots/performance_records.csv"
            df_performance = pd.DataFrame([performance_records])
            df_performance.to_csv(output_path, mode="a", index=False, header=not os.path.exists(output_path))

            print(f"\nH: {self.h}, K: {self.k}, P:{self.p}, SIGMA: {self.sigma}\n"
                  f"DISTORTION: {distortion}, TOTAL TIME: {performance_records['time_total']}, "
                  f"TIME FIND MIN-MOLES: {performance_records['time_find_min_moles']}\n")

            print("\nPreparing the anonymized file")
            with open(r'Dataset/Anonymized/anonymized.txt', 'w') as f:
                for t in self.dataset:
                    transaction = ' '.join(map(str, t))
                    f.write(f"{transaction}\n")
                print('Done')
