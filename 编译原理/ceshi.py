from lexical_analysis import Scanner
selected_fa_files = ['data/DFA/IDN.json', 'data/DFA/keyWord.json', 'data/DFA/number.json', 'data/DFA/operator.json']
f = open('data/test_code/code.c')
code = f.read()
lexical = Scanner(selected_fa_files, code)
Tokens = lexical.analysis()
for token in Tokens:
    print(token)