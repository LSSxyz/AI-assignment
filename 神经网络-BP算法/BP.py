import wx
import os
from 数据获取 import *
from train import train_and_test
class BP(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,None,title='BP算法-函数拟合',size=(640,480))
        self.SelBtn = wx.Button(self,label='浏览',pos=(245,5),size=(80,25))
        self.SelBtn.Bind(wx.EVT_BUTTON,self.OnOpenFile)
        self.OkBtn = wx.Button(self,label='显示数据',pos=(345,5),size=(80,25))
        self.OkBtn.Bind(wx.EVT_BUTTON,self.ReadFile)
        self.DeleteBtn = wx.Button(self, label='取消',pos=(545, 5), size=(60,25))
        self.DeleteBtn.Bind(wx.EVT_BUTTON, self.Delete)
        self.ListBtn = wx.Button(self, label='选择网络结构', pos=(445, 5), size=(90, 25))
        self.ListBtn.Bind(wx.EVT_BUTTON, self.Select)
        self.FileName = wx.TextCtrl(self,pos=(5,5),size=(230,25))
        self.FileContent = wx.TextCtrl(self,pos=(5,35),size=(620,480),style=(wx.TE_MULTILINE))

        self.selection = ""
        
    def OnOpenFile(self,event):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None,'select',os.getcwd(),'',wildcard,wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            # print(dialog.GetPath())
            self.FileName.SetValue(dialog.GetPath())
            dialog.Destroy
            
    def ReadFile(self,event):
        file = open(self.FileName.GetValue())  
        self.FileContent.SetValue(file.read())  
        file.close()  

    def Select(self, event):

        dlg = wx.SingleChoiceDialog(None, u"请选择:", u"选择列表",
                                    [u"单入单出", u"多入单出", ])
        if dlg.ShowModal() == wx.ID_OK:
            self.selection = dlg.GetStringSelection() #获取选择的内容
            dlg_tip = wx.MessageDialog(None, self.selection, u"选择列表", wx.OK | wx.ICON_INFORMATION)
            if self.selection == '单入单出':
                if self.FileName.GetValue()=='':
                    ans = wx.MessageDialog(None, "请先选择文件", u"提示", wx.OK | wx.ICON_INFORMATION)
                    if ans.ShowModal()==wx.ID_OK:
                        ans.Destroy()
                else:
                    Show_single(self.FileName.GetValue())
                    self.AddLayer(self.selection)
                    
            elif self.selection == '多入单出':
                if self.FileName.GetValue()=='':
                    ans = wx.MessageDialog(None, "请先选择文件", u"提示", wx.OK | wx.ICON_INFORMATION)
                    if ans.ShowModal()==wx.ID_OK:
                        ans.Destroy()
                else:
                    Show_multiple(self.FileName.GetValue())
                    self.AddLayer(self.selection)
            # else:
            #     if self.FileName.GetValue()=='':
            #         ans = wx.MessageDialog(None, "请先选择文件", u"提示", wx.OK | wx.ICON_INFORMATION)
            #         if ans.ShowModal()==wx.ID_OK:
            #             ans.Destroy()
            #     else:
            #         Show_multiple(self.FileName.GetValue())
            #         self.AddLayer(self.selection)

            dlg_tip.Destroy()
        dlg.Destroy()
    # 动态添加网络结构
    def AddLayer(self, selection):
        dlg = wx.TextEntryDialog(None, u"请按序网络详细结构(空格间隔):", u"神经网络BP算法", u"示例：2 tanh 3 sigmoid 5 relu 1")
        mes1 = ""
        if dlg.ShowModal() == wx.ID_OK:
            mes1 = dlg.GetValue() #获取文本框中输入的值
            mes1 = mes1.strip()
            # 判断输入是否合法
            if selection=='单入单出':
                if mes1[0] != '1' or mes1[-1]!='1':
                    tip = wx.MessageDialog(None, "输入层和输出层结点个数必须为1", u"提示", wx.OK | wx.ICON_INFORMATION)
                    if tip.ShowModal()==wx.ID_OK:
                        tip.Destroy()
                    self.AddLayer(selection)
                else:
                    self.Input_lr_and_epoch(mes1, selection)
            elif selection=='多入单出':
                if mes1[-1]!='1':
                    tip = wx.MessageDialog(None, "输出层结点个数必须为1", u"提示", wx.OK | wx.ICON_INFORMATION)
                    if tip.ShowModal()==wx.ID_OK:
                        tip.Destroy()
                    self.AddLayer(selection)
                else:
                    self.Input_lr_and_epoch(mes1, selection)
            else:
                # 输入学习率和迭代次数
                self.Input_lr_and_epoch(mes1, selection)
        dlg.Destroy() 

    def Input_lr_and_epoch(self, mes1, selection):
        dlg = wx.TextEntryDialog(None, u"请输入学习率和迭代次数(空格间隔):", u"神经网络BP算法", u"示例：0.1 100")
        mes2 = ""
        if dlg.ShowModal() == wx.ID_OK:
            mes2 = dlg.GetValue() #获取文本框中输入的值
            train_and_test(mes1, mes2, selection, self.FileName.GetValue())
    # 清空已选择的数据
    def Delete(self, event):
        self.FileName.SetValue("")
        self.FileContent.SetValue("")
if __name__=='__main__':
    app = wx.App()
    SiteFrame = BP()
    SiteFrame.Show()
    app.MainLoop()