import wx
from GUI import GUI

if __name__ == "__main__":
    app = wx.App()
    frame = GUI(None, 'Lexer')
    app.MainLoop()
