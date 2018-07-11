# coding:utf-8
# 这个模块用于绘制构造好的决策树
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_mid_text(cntr_pt, parent_pt, text):
    """
    cntr_pt: 子节点的坐标
    parent_pt: 父节点的坐标
    text: 连线上的文字
    这个函数的功能就是给连线加上文字
    """
    x_mid = (parent_pt[0] - cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, text)

def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    node_text: str, 节点中的文字
    center_pt: 节点的中心坐标
    parent_pt: 父节点的坐标
    node_type: 节点的类型，上面定义的decision_node或者leaf_node
    """
    create_plot.ax1.annotate(node_text, xy=parent_pt,
        xycoords='axes fraction', xytext=center_pt, 
        textcoords='axes fraction', va="center",
        ha="center", bbox=node_type, arrowprops=arrow_args)

def get_num_leafs(my_tree):
    """
    my_tree: dict, 算法所生成的树，字典的格式
    这个函数确定一共有几个叶子节点
    """
    num_leafs = 0
    # 得到第一个用于划分的特征的名字
    first_str = list(my_tree.keys())[0]
    # 得到这个特征作为根节点的树
    second_dict = my_tree[first_str]
    # 遍历这个树，如果其中还有分支的话就递归遍历
    # 如果没有分支的话就加一，加的一代表目前这个树的根节点是一个叶子节点
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    """
    my_tree: dict, 算法生成的树，字典格式
    这个函数确定生成的树的深度
    """
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    # 深度优先遍历，记录每条路径的深度，当深度更大的时候借更新最大深度
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def retrieve_tree(i):
    """
    这个函数返回已经构造好的树，用于测试
    """
    list_of_tree = [{'no surfacing': {0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}}, {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return list_of_tree[i]

def create_plot(in_tree):
    """
    in_tree: dict, 字典格式的树
    这个函数是绘制树的主函数
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()

def plot_tree(my_tree, parent_pt, node_text):
    """
    my_tree: dict,算法生成的树，字典格式
    parent_pt: tuple,二维的元祖，父节点的坐标
    node_text: str,节点上的文字
    """
    # 先得到树的深度和叶子节点的数目
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    # 得到第一个节点的名字
    first_str = list(my_tree.keys())[0]
    # 根据叶子节点数和深度计算第一个节点的位置
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)
    # 绘制其与子节点之间的文字
    plot_mid_text(cntr_pt, parent_pt, node_text)
    # 绘制这个节点
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    # 得到这个节点的子节点
    second_dict = my_tree[first_str]
    # 确定子节点的y轴位置
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
    # 如果子节点是是一个字典，表示还有孙节点，递归绘制树
    # 如果子节点不是一个字典，那么就到这里为止，直接绘制节点和文字
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff),
                    cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD

if __name__ == '__main__':
    #create_plot()
    my_tree=retrieve_tree(0)
    create_plot(my_tree)