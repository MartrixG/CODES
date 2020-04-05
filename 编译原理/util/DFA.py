import string


class DFA(object):
    def __init__(self, data):
        self.name = data['name'][0]
        self.Q = data['Q'][0].split(' ')
        self.sigma = data['sigma'][0].split(' ')
        self.q0 = data['q0'][0]
        species = data['F'][0].split(' ')
        self.F = {}
        for spec in species:
            tmp = spec.split(":")
            self.F[tmp[0]]=tmp[1]
        tmp = ""
        for item in data['t']:
            tmp += item
        t = {}
        for item in tmp.split('|'):
            key, value = item.split(',')
            key = [key]
            if key[0] == 'digit':
                key = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            if key[0] == 'letter':
                key = [i for i in string.ascii_letters]
            for each_key in key:
                t[value[0] + each_key] = value[1]
        self.t = t

    def scan(self, pos, content):
        re = ''
        now_state = self.q0
        while True:
            now_ch = content[pos]
            if self.t.get(now_state+now_ch, None) is None:
                if now_state in self.F.keys():
                    return re, self.F.get(now_state), pos
                else:
                    return -1, now_state+now_ch, -1
            else:
                now_state = self.t.get(now_state+now_ch)
                re += now_ch
                pos += 1

    def get_list(self):
        re = []
        line = ["s\\Q"]
        for s in self.Q:
            line.append(s)
        re.append(line)
        for s in self.sigma:
            line = [s]
            for to in self.Q:
                tmp = to + s
                if self.t.get(tmp, None) is None:
                    line.append('err')
                else:
                    line.append(self.t.get(tmp))
            re.append(line)
        return re

    def __repr__(self):
        re = self.name + ":\n" + "s\\Q\t"
        for s in self.Q:
            re += s+'\t'
        re += '\n'
        for s in self.sigma:
            re += s+'\t'
            for to in self.Q:
                tmp = to + s
                if self.t.get(tmp, None) is None:
                    re += 'err\t'
                else:
                    re += self.t.get(tmp) + '\t'
            re += '\n'
        return re