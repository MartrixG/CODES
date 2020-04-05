import os
import wx
import json
from tabulate import tabulate
from lexical_analysis import Scanner


class GUI(wx.Frame):
    def __init__(self, parent, title):
        super(GUI, self).__init__(parent, title=title, size=(1500, 600))
        self.InitUI()
        self.Centre()
        self.Show()

    def InitUI(self):
        default_font = wx.Font(10, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Consolas')
        panel = wx.Panel(self)
        sizer = wx.GridBagSizer(0, 0)

        text_input = wx.StaticText(panel, label="Input")
        sizer.Add(text_input, pos=(0, 0), flag=wx.ALL, border=5)
        self.input_box = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_PROCESS_ENTER)
        self.input_box.SetFont(default_font)
        sizer.Add(self.input_box, pos=(0, 1), span=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        text_output = wx.StaticText(panel, label="Output")
        sizer.Add(text_output, pos=(0, 2), flag=wx.ALL, border=5)
        self.output_box = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL)
        sizer.Add(self.output_box, pos=(0, 3), span=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.output_box.SetFont(default_font)

        text_output_fa = wx.StaticText(panel, label="DFA Output")
        sizer.Add(text_output_fa, pos=(0, 4), flag=wx.ALL, border=5)
        self.output_fa_box = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.VSCROLL | wx.HSCROLL)
        self.output_fa_box.SetFont(default_font)
        sizer.Add(self.output_fa_box, pos=(0, 5), span=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        text_fa_list = wx.StaticText(panel, label="FA Files")
        sizer.Add(text_fa_list, pos=(0, 6), flag=wx.ALL, border=5)
        self.fa_list_box = wx.ListBox(panel, style=wx.LB_MULTIPLE | wx.LB_HSCROLL)
        self.fa_list_box.SetFont(default_font)
        sizer.Add(self.fa_list_box, pos=(0, 7), span=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)

        sizer.AddGrowableRow(1)
        sizer.AddGrowableCol(1)
        sizer.AddGrowableCol(3)
        sizer.AddGrowableCol(5)
        sizer.AddGrowableCol(7)

        buttonSelectCodeFile = wx.Button(panel, label="Select Code File")
        buttonSelectCodeFile.Bind(wx.EVT_BUTTON, self.OnPressSelectCodeFileBtn)
        sizer.Add(buttonSelectCodeFile, pos=(2, 0), span=(1, 2), flag=wx.ALL, border=5)

        buttonSelectFaFile = wx.Button(panel, label="Select FA File")
        buttonSelectFaFile.Bind(wx.EVT_BUTTON, self.OnPressSelectFaFileBtn)
        sizer.Add(buttonSelectFaFile, pos=(2, 2), span=(1, 2), flag=wx.ALL, border=5)

        buttonProcess = wx.Button(panel, label="Process")
        buttonProcess.Bind(wx.EVT_BUTTON, self.OnPressProcessBtn)
        sizer.Add(buttonProcess, pos=(2, 4), span=(1, 2), flag=wx.ALL, border=5)

        panel.SetSizerAndFit(sizer)

    def OnPressSelectCodeFileBtn(self, event):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            dialog.Destroy
            try:
                f_in = open(file_path, 'r', encoding='utf-8')
                self.code_file = file_path
                self.input_box.SetValue(''.join(f_in.readlines()))
            except:
                wx.MessageBox("Cannot load file", "Error", wx.OK | wx.ICON_ERROR)

    def OnPressSelectFaFileBtn(self, event):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)
        if dialog.ShowModal() == wx.ID_OK:
            file_paths = dialog.GetPaths()
            dialog.Destroy
            try:
                for file_path in file_paths:
                    f_in = open(file_path, 'r', encoding='utf-8')
                    try:
                        json.loads(f_in.read())
                        self.fa_list_box.Append(os.path.relpath(file_path, os.getcwd()))
                    except json.decoder.JSONDecodeError:
                        wx.MessageBox("Not json data!\nPlease select another one!", "Error", wx.OK | wx.ICON_ERROR)
                    except:
                        wx.MessageBox("Unknown Error!", "Error", wx.OK | wx.ICON_ERROR)
            except:
                wx.MessageBox("Cannot load file", "Error", wx.OK | wx.ICON_ERROR)

    def OnPressProcessBtn(self, event):
        selected_fa_files = [self.fa_list_box.GetString(e) for e in self.fa_list_box.GetSelections()]
        lexical = Scanner(selected_fa_files, self.input_box.GetValue())
        Tokens = lexical.analysis()
        tokens = []
        for token in Tokens:
            tokens.append(token.__repr__())
        self.output_box.SetValue('\n'.join(tokens))
        output_fa_text = ""
        for dfa_name, dfa in lexical.dfas:
            output_fa_text += '{}\n'.format(dfa_name)
            dfa_table = dfa.get_list()
            output_fa_text += tabulate(dfa_table[1:], dfa_table[0], tablefmt="grid") + '\n'
        self.output_fa_box.SetValue(output_fa_text)
