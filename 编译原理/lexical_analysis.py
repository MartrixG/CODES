from util.DFA import DFA
import json


def num_error(state):
    token = []
    if state[0] == 'D':
        token.append("LexicalError! There must be 0-9 or A-E numbers after the hexadecimal number.")
    elif state[0] == 'F':
        token.append("LexicalError! There must be a number after the decimal point to represent the float.")
    elif state[0] == 'H':
        token.append("LexicalError! There must be a number after E to indicate scientific notation.")
    elif state[0] == 'I':
        token.append("LexicalError! There must be a number after the + and-signs.")
    return token


class Scanner(object):
    def __init__(self, json_files, codes):
        self.keyWord = {}
        self.operator = {}
        self.dfas = []
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
            if data['name'][0] == 'number':
                self.numDFA = DFA(data)
                self.dfas.append(["numbers DFA", self.numDFA])
            elif data['name'][0] == 'IDN':
                self.idnDFA = DFA(data)
                self.dfas.append(["Identifiers DFA", self.idnDFA])
            elif data['name'][0] == 'keyWord':
                keyWord = data['keys'][0].split(' ')
            elif data['name'][0] == 'operator':
                operator = data['keys'][0].split(' ')
            else:
                raise SyntaxError()
        # print(self.numDFA.F)
        # print(self.idnDFA.F)
        self.codes = codes
        for i in range(len(keyWord)):
            self.keyWord[keyWord[i]] = i
        for i in range(len(operator)):
            self.operator[operator[i]] = i+len(keyWord)

    def analysis(self):
        token = []
        pos = 0
        while pos != len(self.codes):
            now_ch = self.codes[pos]
            if now_ch in (' ', '\n', '\t'):
                pos += 1
            elif '0' <= now_ch <= '9':
                re, spec, pos = self.numDFA.scan(pos, self.codes)
                if pos != -1:
                    token.append("{:}\t<{:},{:}>".format(re, spec, re))
                else:
                    token = num_error(spec)
                    return token
            elif now_ch in self.operator.keys():
                tmp_operator = now_ch
                if pos + 1 != len(self.codes):
                    if now_ch + self.codes[pos + 1] in self.operator.keys():
                        tmp_operator += self.codes[pos + 1]
                        pos += 1
                pos += 1
                token.append("{:}\t<{:},{:}>".format(tmp_operator, self.operator.get(tmp_operator), ' _ '))
            elif now_ch.isalpha() or now_ch == '_':
                re, spec, pos = self.idnDFA.scan(pos, self.codes)
                if re in self.keyWord:
                    token.append("{:}\t<{:}, {:} >".format(re, re.upper(), "_"))
                else:
                    token.append("{:}\t<{:}, {:} >".format(re, "IDN", re))
        return token
