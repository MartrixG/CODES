import wx
from UI.GUI import GUI
from complier.lexical_analysis import Scanner

if __name__ == "__main__":
    # app = wx.App()
    # frame = GUI(None, 'Lexer')
    # app.MainLoop()
    selected_fa_files = ['data/DFA/IDN.json', 'data/DFA/keyWord.json', 'data/DFA/number.json', 'data/DFA/operator.json']
    f = open('data/test_code/code.c')
    code = f.read()
    lexical = Scanner(selected_fa_files, code)
    Tokens, errors = lexical.analysis()
    for token in Tokens:
        print(token)
