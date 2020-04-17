import json

from util.element import element
from util.produnction import make_production, make_item, recover_item

EPSILON = element('#')
DOLLAR = element('$')


class LR(object):
    def __init__(self, path):
        self.all_prod = []
        self.closures = {}
        self.alphabet = set()
        self.first = {}
        self.producers = {}
        self.start = None
        self.goto = None
        self.action = None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for left in data.keys():
                if self.start is None:
                    self.all_prod += make_production(left + "_p", [left])
                    self.start = self.all_prod[0].left
                self.all_prod += make_production(left, data[left])
        for prod in self.all_prod:
            self.alphabet.add(prod.left)
            for rights in prod.right:
                self.alphabet.add(rights)
            if prod.left not in self.producers:
                self.producers[prod.left] = set()
            self.producers[prod.left].add(prod)
        self.alphabet.add(DOLLAR)
        self.init_first()
        self.init_closure()
        self.init_goto_action()

    def get_closure(self, start_I):
        closure = set(start_I)
        while True:
            tmp_closure = set()
            for item in closure:
                right = item.right
                dot_pos = item.dot_pos
                forward = item.forward
                if dot_pos >= len(right) or right[dot_pos].type != "VN":  # 遇到终结符或者末尾直接跳过
                    continue
                if dot_pos < len(right) and right[dot_pos].type == "VN":
                    beta = right[dot_pos + 1:]  # 非终结符之后所有element均为beta
                    first = self.get_first(beta + [forward])  # beta + 展望符 求first
                    B = right[dot_pos]  # 圆点之后的第一个符号
                    for prod in self.producers[B]:  # 对所有这个非终结符的产生式进行迭代
                        for b in first:  # 对first集所有的符号迭代
                            if b == EPSILON:  # 如果first集中出现了空串报错
                                print(beta + forward)
                                raise Exception("在beta-a产生式的first集中出现了空产生式")
                            pos = 0
                            while pos < len(prod.right) and prod.right[pos] == EPSILON:  # 预防万一跳过空串开头
                                pos += 1
                            tmp_item = make_item(prod, pos, b)
                            if tmp_item not in closure:
                                tmp_closure.add(tmp_item)
            if tmp_closure == set():
                break
            else:
                closure |= tmp_closure
        return frozenset(closure)

    def go(self, closure_I, X):
        next_closure = set()
        for item in closure_I:
            if item.dot_pos < len(item.right) and item.right[item.dot_pos] == X:
                from copy import deepcopy
                tmp_item = deepcopy(item)
                tmp_item.dot_pos += 1
                next_closure.add(tmp_item)
        return self.get_closure(list(next_closure))

    def init_goto_action(self):
        for prod in self.producers[self.start]:
            start_prod = prod
        self.goto = {}
        self.action = {}
        for state in self.closures:
            self.goto[state] = {}
            self.action[state] = {}
            for ele in self.alphabet:
                if ele.type == 'VN':
                    tmp_goto = self.go(state, ele)
                    if tmp_goto in self.closures:
                        self.goto[state][ele] = tmp_goto
            for prod in state:
                if prod.dot_pos >= len(prod.right):
                    self.action[state][prod.forward] = ('r', recover_item(prod))
                else:
                    a = prod.right[prod.dot_pos]
                    if a == EPSILON:
                        raise Exception("创建goto转移表时出现空产生式错误")
                    if a.type == 'VT':
                        self.action[state][a] = ('s', self.go(state, a))
                if start_prod.product_eq(prod) and prod.dot_pos == 1 and prod.forward == DOLLAR:
                    self.action[state][DOLLAR] = ('acc', 0)

    def init_closure(self):
        for prod in self.producers[self.start]:
            start_prod = prod
        start_I = make_item(start_prod, 0, DOLLAR)
        self.closures[self.get_closure([start_I])] = 0
        while True:
            D = set()
            for closure in self.closures.keys():
                for X in self.alphabet:
                    if X != EPSILON:
                        tmp_closure = self.go(closure, X)
                        if len(tmp_closure) != 0 and tmp_closure not in self.closures:
                            D.add(tmp_closure)
            if len(D) == 0:
                break
            else:
                for d in D:
                    self.closures[d] = len(self.closures)

    def get_first(self, prod):
        first_set = set()
        have_epsilon = True
        for ele in prod:
            if not have_epsilon:
                break
            first_set |= self.first[ele]
            if EPSILON not in self.first[ele]:
                have_epsilon = False
        if have_epsilon:
            first_set.add(EPSILON)
        elif EPSILON in first_set:
            first_set.remove(EPSILON)
        return first_set

    def init_first(self):
        for elements in self.alphabet:
            if elements.type != 'VN':
                self.first[elements] = {elements}
            else:
                self.first[elements] = set()
        updated = True
        while updated:
            updated = False
            for prod in self.all_prod:
                tmp = self.get_first(prod.right)
                if not self.first[prod.left].issuperset(tmp):
                    updated = True
                self.first[prod.left] |= tmp

    def print_productions(self):
        for prod in self.all_prod:
            print(prod)

    def print_alphabet(self):
        for elements in self.alphabet:
            print(elements)

    def print_first(self):
        for ele in self.alphabet:
            if ele.type == 'VN':
                print(ele, self.first[ele])

    def print_closure(self):
        for item in self.closures.keys():
            print(self.closures[item])
            print('-' * 30)
            for prod in item:
                print(prod)


if __name__ == '__main__':
    grammar_lr = LR('../data/grammar/c_style.json')
    grammar_lr.print_closure()
