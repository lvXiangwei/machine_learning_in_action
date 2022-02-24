import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth',fc='0.8')
leaf_node = dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')

def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt,
    xycoords='axes fraction',
    xytext=center_pt, textcoords='axes fraction',
    va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.savefig('test_plot.png')
    plt.show()

# need to know the number of leaf node to size things in X axis
# need to know depth to size y axis

def get_number_leafs(my_tree):
    if type(my_tree) != dict: return 1
    root_key = list(my_tree.keys())[0]
    subtrees = my_tree[root_key]
    num_leafs = 0
    for key in subtrees:
        num_leafs+=get_number_leafs(subtrees[key])
    return num_leafs

def get_tree_depth(my_tree):
    if type(my_tree) != dict: return 0
    
    root_key = list(my_tree.keys())[0]
    subtrees=my_tree[root_key]
    max_subtree_depth = 0
    for key in subtrees:
        subtree_depth = get_tree_depth(subtrees[key])
        if subtree_depth > max_subtree_depth:
            max_subtree_depth = subtree_depth
    return 1+max_subtree_depth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
    {0: 'no', 1: 'yes'}}}},
    {'no surfacing': {0: 'no', 1: {'flippers': \
    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]