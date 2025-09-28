import json
from collections import deque

class SequenceTreeNode:
    __slots__ = ['value', 'parent', 'children', 'is_leaf', 'flag','count']
    
    def __init__(self, value, parent=None):
        self.value = value      # 当前节点存储的字符串
        self.parent = parent    # 父节点引用
        self.children = {}      # 子节点字典 {child_value: node}
        self.is_leaf = False    # 是否为叶子节点
        self.flag = False       # 叶子节点记录是否异常
        self.count = 0

class SequenceTree:
    def __init__(self, seq_length=100):
        self.root = SequenceTreeNode(None)
        self.seq_length = seq_length
    
    def _traverse(self, sequence):
        """安全遍历路径节点，返回最终节点"""
        if len(sequence) != self.seq_length:
            raise ValueError(f"序列长度必须为{self.seq_length},当前节点长度为",len(sequence))
            
        current = self.root
        for s in sequence:
            if s not in current.children:
                return None
            current = current.children[s]
        if current.is_leaf: 
            current.count +=1
            return current
        else: return None


    def insert(self, sequence, flag=True):
        """插入序列并设置标记"""
        if len(sequence) != self.seq_length:
            print(sequence,len(sequence))
            raise ValueError(f"序列长度必须为{self.seq_length},当前节点长度为",len(sequence))
            
        current = self.root

        if self._traverse(sequence) : return True

        for depth, s in enumerate(sequence):
            # 自动创建路径节点
            if s not in current.children:
                current.children[s] = SequenceTreeNode(s, current)
                
            current = current.children[s]
            
            # 到达末端时设置叶子标记
            if depth == self.seq_length - 1:
                current.is_leaf = True
                current.count=current.count+1
                current.flag = flag

    def modify(self, sequence, new_flag):
        """修改已有序列的标记"""
        node = self._traverse(sequence)
        if node and node.is_leaf:
            node.flag = new_flag
            node.count+=1
            return True
        return False

    def contains(self, sequence):
        """检查序列是否存在"""
        node = self._traverse(sequence)
        return node is not None

    def save(self, filename):
        """广度优先序列化存储"""
        data = {
            "meta": {"seq_length": self.seq_length},
            "nodes": []
        }
        
        # 生成节点ID映射
        node_id = {id(None): 0}
        queue = deque([(self.root, 0)])
        while queue:
            node, parent_id = queue.popleft()
            current_id = len(data["nodes"]) + 1
            node_id[id(node)] = current_id
            
            node_data = {
                "value": node.value,
                "parent": parent_id,
                "children": [],
                "is_leaf": node.is_leaf,
                "flag": node.flag,
                "count": node.count
            }
            
            # 记录子节点关系
            for child in node.children.values():
                node_data["children"].append(child.value)
                queue.append((child, current_id))
                
            data["nodes"].append(node_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename):
        """从文件加载树结构"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        tree = cls(data["meta"]["seq_length"])
        nodes = {0: None}  # ID到节点的映射
        
        # 第一遍创建所有节点
        for node_data in data["nodes"]:
            parent = nodes.get(node_data["parent"])
            node = SequenceTreeNode(
                value=node_data["value"],
                parent=parent
            )
            node.is_leaf = node_data["is_leaf"]
            node.flag = node_data["flag"]
            node.count = node_data["count"]
            nodes[node_data["parent"]] = parent
            nodes[len(nodes)] = node  # 按顺序分配ID
            
        # 第二遍建立子节点引用
        for idx, node_data in enumerate(data["nodes"]):
            if idx == 0:
                tree.root = nodes[1]
                continue
                
            node = nodes[idx+1]
            if node.parent:
                node.parent.children[node.value] = node
                
        return tree


import pandas as pd

# # 使用示例
# if __name__ == "__main__":
#     # 初始化树
#     tree = SequenceTree(seq_length=20)  # 测试使用短长度
    
#     df = pd.read_csv('/home/hanchunhui/projects/hhr/rlhf/code/window_ad/BGL/20l_bgl.csv')
#     for i in range(len(df)):
#         list_=eval(df['EventList'][i])
#         if df['Status'][i]=="success" :flag_=True
#         else: flag_=False

#         # 插入序列
#         tree.insert(list_, flag=flag_)
        
#         # # 修改标记
#         # tree.modify(seq1, False)
    
#     # 保存与加载
#     tree.save("sequence20_tree_ad_50000.json")
#     loaded_tree = SequenceTree.load("sequence20_tree_ad_50000.json")
#     print(eval(df['EventList'][0]),len(eval(df['EventList'][0])))
#     print("原始树查询:", tree.contains(eval(df['EventList'][0])))  # True
#     print("加载树查询:", loaded_tree.contains(eval(df['EventList'][0])))  # True
#     print("原始树查询:", tree.contains(eval(df['EventList'][0])))  # True
#     print("加载树查询:", loaded_tree.contains(eval(df['EventList'][0])))  # True
#     print("标记状态:", loaded_tree._traverse(eval(df['EventList'][0])).flag, loaded_tree._traverse(eval(df['EventList'][0])).count)  # False

# python build_tree.py


