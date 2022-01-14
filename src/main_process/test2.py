import os
from typing import List, Dict, Union
from treelib import Tree, Node
import math
import cassis
texts = ["Dramatische Wende im Ringen um die finanziell angeschlagenen MV Werften",
         "Die Geschäftsführung hat in einer internen Mitteilung einen Insolvenzantrag angekündigt.",
         "Das Amtsgericht Schwerin bestätigte den Eingang des Antrags.",
         "Am Nachmittag berät der Finanzausschuss des Landes in einer Sondersitzung darüber",
         "wie es weitergeht. Um 14.15",
         "Uhr berichten wir in einem NDR MV Live.",
         "Joe waited for the train . The train was late .",
         "Ich bin ein dummer Text!!!"]
casses = []
for i in range(0, 7):
    cas = cassis.Cas()
    cas.sofa_string = texts[i]
    cas.sofa_mime = "text/plain"
    casses.append(cas)
    del cas




tree = Tree()
tree.create_node("Harry", "harry", data=casses[0])  # root node
tree.create_node("Jane", "jane", parent="harry", data=casses[1])
tree.create_node("Bill", "bill", parent="harry", data=casses[2])
tree.create_node("Diane", "diane", parent="jane", data=casses[3])
tree.create_node("Mary", "mary", parent="diane", data=casses[4])
tree.create_node("Mark", "mark", parent="jane", data=casses[5])
tree.create_node("Leon", 0, parent="bill", data=casses[6])


def node_path_dist(node1: Node,
                   node2: Node,
                   tree: Tree) -> Union[int, float]:
    """
    Function calculates the geodesic distance for two nodes in a
    given dependency tree. If nodes are not connected in the tree, distance
    becomes +inf.
    :param node1:
    :param node2:
    :param tree:
    :return:
    """
    name1 = node1.identifier
    name2 = node2.identifier
    if tree.is_ancestor(name1, name2):
        return tree.level(name2) - tree.level(name1)
    elif tree.is_ancestor(name2, name1):
        # return tree.level(name1) - tree.level(name2)
        return math.inf
    else:
        return math.inf


def make_dist_dict(_Vs: List[Node],
                   dep_tree: Tree) ->  Dict[Node, Dict[Union[int, float], List[Node]]]:
    """
    Function for creating a dict containing every path distances for all vertices
    to all other vertices.
    Dict keys are Nodes, and values are dicts. For these dicts keys are int or float and value
    list of nodes.
    :param _Vs:
    :param dep_tree:
    :return:
    """
    dist_dict = {}
    for i in _Vs:
        a = []
        for j in _Vs:
            a.append((node_path_dist(i, j, dep_tree), j))
        b = dict()
        for j in a:
            if j[0] in b:
                b[j[0]].append(j[-1])
            else:
                b[j[0]] = [j[-1]]
        dist_dict[i] = b
    return dist_dict

def out_degree(node: Node, dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]]) -> int:
    """
    Function returns the out-degree of a given node in a dependency-tree.
    :param node:
    :param dist_dict:
    :return:
    """
    try:
        return len(dist_dict[node][1])
    except:
        return 0


def get_prestige(dist_dict: Dict[Node, Dict[Union[int, float], List[Node]]]):
    """
    Function for calculating the absolute prestige:
    https://www.researchgate.net/publication/200110853_Structural_analysis_of_hypertexts_Identifying_hierarchies_and_useful_

    :param dist_dict:
    :return:
    """
    node_prestiges = []
    for node in dist_dict:
        status = 0
        for dist in dist_dict[node]:
            if 0 < dist < math.inf:
                status += (dist * len(dist_dict[node][dist]))
        contra_status = 0
        for node2 in dist_dict:
            for dist2 in dist_dict[node2]:
                if node in dist_dict[node2][dist2]:
                    if 0 < dist2 < math.inf:
                        contra_status += dist2
                    break
        prestige_of_node = status - contra_status
        node_prestiges.append(abs(prestige_of_node))
    absolute_prestige = sum(node_prestiges)
    return absolute_prestige

def get_ssl(dist_dict, root_node, _L):
    ssl = dict()
    for dist in dist_dict[root_node]:
        leaves = []
        for leave in _L:
            if leave in dist_dict[root_node][dist]:
                leaves.append(leave)
        if leaves:
            ssl[dist] = leaves
    return ssl

def get_lde(ssl: Dict[Union[int, float], List[Node]],
            _L: List[Node]):
    """
    Function for calculating LDE (Leaf Depth/Distance Entropy)
    :param ssl:
    :param _L:
    :return:
    """
    total_number_of_subsets = len(ssl.keys())
    if total_number_of_subsets > 1:
        lde_sum = 0
        for dist in ssl:
            pi = len(ssl[dist]) / len(_L)
            lde_sum += (pi * math.log2(pi))
        lde_sum *= -1
        lde = lde_sum / math.log2(total_number_of_subsets)
        return lde
    else:
        return 0


def get_highest_level_predecessor(node1: Node, node2: Node, dep_tree: Tree) -> Node:
    """
    Function for finding highest level predecessor for two
    given nodes in their tree.
    :param node1:
    :param node2:
    :param dep_tree:
    :return:
    """
    tree_id = dep_tree.identifier
    predecessors_node1 = []
    if node1.is_root(tree_id) or node2.is_root(tree_id):
        return tree.get_node(tree.root)
    else:
        while True:
            predecessor = node1.predecessor(tree_id)
            if predecessor is None:
                break
            else:
                predecessors_node1.append(predecessor)
                node1 = tree.get_node(predecessor)

        predecessors_both = []
        while True:
            predecessor = node2.predecessor(tree_id)
            if predecessor is None:
                break
            else:
                if predecessor in predecessors_node1:
                    predecessors_both.append(predecessor)
                node2 = tree.get_node(predecessor)

        final_predecessor = ("", -1)
        for predecessor in predecessors_both:
            depth = dep_tree.depth(predecessor)
            if depth > final_predecessor[-1]:
                final_predecessor = (predecessor, depth)

        return tree.get_node(final_predecessor[0])




tree.show()
dist_dict = make_dist_dict(tree.all_nodes(), tree)
"""
print(get_prestige(dist_dict))
ssl = get_ssl(dist_dict, tree.get_node(tree.root), tree.leaves())
print(ssl)
print(get_lde(ssl, tree.leaves()))

print(get_highest_level_predecessor(tree.get_node("harry"), tree.get_node("jane"), tree).identifier)
#print(tree.get_node("harry").predecessor(tree.identifier))
test_list = tree.leaves()
leave_pairs = [(a.identifier, b.identifier) for idx, a in enumerate(test_list) for b in test_list[idx + 1:]]
print(test_list)
print(leave_pairs)
"""
for i in dist_dict:
    print(tree.level(i.identifier), dist_dict[i])
print(tree.depth())