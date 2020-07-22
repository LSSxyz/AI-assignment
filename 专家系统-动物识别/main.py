import wx 
from 正向推理 import Inference, Get_G
from 逆向推理 import backward_inference
class MyPanel(wx.Panel): 
    def __init__(self, parent, id): 
        wx.Panel.__init__(self, parent, id) 
        try: 
            image_file = 'panel背景.png' 
            to_bmp_image = wx.Image(image_file, wx.BITMAP_TYPE_ANY).ConvertToBitmap() 
            self.bitmap = wx.StaticBitmap(self, -1, to_bmp_image, (0, 0)) 
            image_width = to_bmp_image.GetWidth() 
            image_height = to_bmp_image.GetHeight() 
            set_title = "动物识别专家系统"
            parent.SetTitle(set_title) 
        except IOError: 
            print('error')
            # raise SystemExit 
  #创建一个按钮 
        try:
            btn_image = 'button.png'
            btn = wx.Image(btn_image, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            self.button = wx.BitmapButton(self.bitmap, -1, btn, pos=(300,350)) 
        except IOError:
            print("error")
        
        self.Bind(wx.EVT_BUTTON, self.Select, self.button)
        self.message = ""
        self.selection = ""
    def process(self, ):
        input_list = self.message.split(' ')
        if self.selection == "正向推理":
            input_his, terminal = Inference().inference(input_list)
            outmess = ""
            indx = 1
            for cell in input_his:
                flag=0
                for item in cell:
                    if flag:
                        outmess = outmess+' '+item
                    else:
                        flag=1
                        outmess = outmess+ '步骤'+str(indx)+' ( '
                        outmess = outmess+item
                        indx+=1
                outmess = outmess+' )'+"\n"

            if terminal!=[['Wrong']]:
                outmess = outmess+"\n"+'推理结果：'+terminal[0]
            else:
                outmess = outmess+"\n"+'条件不充分或者条件错误导致推理失败!'
            # 输出判断集合的变化以及结果
            ans = wx.MessageDialog(None, outmess, u"正向推理", wx.OK | wx.ICON_INFORMATION)
            if ans.ShowModal() == wx.ID_OK:
                ans.Destroy()
        else:
            self.bck(input_list)

    def bck(self, input_list):
        done = [] #已经确认的条件
        infer = backward_inference(input_list[0])
        success=False
        for line in infer:
            if list(set(done).difference(set(line))):continue 

            ask = list(set(line).difference(set(done)))
            cnt=0
            for i in ask:
                dlg_tip = wx.MessageDialog(None, i, u"特征确认", wx.YES_NO | wx.ICON_INFORMATION)
                if dlg_tip.ShowModal() == wx.ID_YES:
                    done.append(i)
                    cnt+=1
                else:
                    break
            if cnt==len(ask):
                success=True
                break
        # 定义对话框输出结果
        outmess = '输入条件：'+input_list[0]+ "\n"+'已确认特征：'
        flag = 0
        for item in done:
            if flag:
                outmess = outmess + ' ' + item
            else:
                flag = 1
                outmess = outmess + '( ' + item
        if success==False:
            if done:
                outmess = outmess + ' ) ' + "\n" + '推理失败！'
            else:
                outmess = outmess + "\n" + '推理失败！'
        else:outmess = outmess + ' ) ' + "\n" +'推理成功！'

        ans = wx.MessageDialog(None, outmess, u"逆向推理", wx.OK | wx.ICON_INFORMATION)
        if ans.ShowModal()==wx.ID_OK:
            ans.Destroy()
   #文本输入对话框    
    def Input(self,):
        dlg = wx.TextEntryDialog(None, u"请在下面文本框中输入内容:", u"动物识别", u"...")
        if dlg.ShowModal() == wx.ID_OK:
            self.message = dlg.GetValue() #获取文本框中输入的值
            self.process()
        dlg.Destroy()  
    # 帮助窗口
    def Help(self, ):
        factor, res = Get_G()
        outmess = '知识库现有规则：'+"\n"
        for i in range(len(factor)):
            flag=0
            for item in factor[i]:
                if flag:
                    outmess = outmess+' and '+item
                else:
                    flag=1
                    cnt=i+1
                    outmess = outmess+'R'+str(cnt)+': ('+item
            outmess = outmess+')->'+res[i]+"\n"
        ans = wx.MessageDialog(None, outmess, u"更多", wx.OK | wx.ICON_INFORMATION)
        if ans.ShowModal()==wx.ID_OK:
            ans.Destroy()


    def Select(self, event):
        dlg = wx.SingleChoiceDialog(None, u"请选择你要进行的操作:", u"操作列表",
                                    [u"正向推理", u"反向推理", u"更多"])
        if dlg.ShowModal() == wx.ID_OK:
            self.selection = dlg.GetStringSelection() #获取选择的内容
            dlg_tip = wx.MessageDialog(None, self.selection, u"选择操作", wx.OK | wx.ICON_INFORMATION)
            if self.selection != '更多':
                self.Input()
            else:
                self.Help()                
            dlg_tip.Destroy()
        dlg.Destroy()

def tranction():
    app = wx.App() 
    frame = wx.Frame(None, -1, 'Image', size=(780,600)) 
    my_panel = MyPanel(frame, -1)
    frame.Show() 
    app.MainLoop() 
   
if __name__ == '__main__': 
    tranction()