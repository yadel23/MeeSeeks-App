from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pandas as pd
from matplotlib.backend_bases import key_press_handler

import io
import base64


class graph_animation:
    def __init__(self, file_x, file_y, file_z, file_mx, file_my, file_mz):
        self.dfx= pd.read_json(file_x)
        self.dfy= pd.read_json(file_y)
        self.dfz= pd.read_json(file_z)
        self.mdfx= pd.read_json(file_mx)
        self.mdfy= pd.read_json(file_my)
        self.mdfz= pd.read_json(file_mz)
        self.dfx.rename(columns={ self.dfx.columns[0]: "nose" }, inplace = True)
        self.mdfx.rename(columns={ self.mdfx.columns[0]: "nose" }, inplace = True)
        self.current_frame = 0

        starting_nose_point = self.dfx['nose'].iloc[0]
        self.mdfx += (starting_nose_point - self.mdfx['nose'].iloc[0])


        self.fig = plt.figure(figsize=(6,5))

        self.grid = plt.GridSpec(6, 5, wspace =0.3, hspace = 0.8)

        self.ax = self.fig.add_subplot(self.grid[:2, :1], projection='3d')
        self.ax2 = self.fig.add_subplot(self.grid[:2, 1], projection='3d')
        self.ax3 = self.fig.add_subplot(self.grid[:5, :1], projection='3d')
        self.ax4 = self.fig.add_subplot(self.grid[:5, 1], projection='3d')
        self.ax5 = self.fig.add_subplot(self.grid[:, 2:], projection='3d')
        # self.ax5.set_xlim3d(.2, .7)
        # self.ax5.set_ylim3d(0, .8)
        # self.ax5.set_zlim3d(0, 1)
        self.ax.view_init(-90, -90)
        self.ax2.view_init(-90, -90)
        self.ax3.view_init(-90, -90)
        self.ax4.view_init(-90, -90)
        self.ax5.view_init(-90, -90)
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_zticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_xticklabels([])
        self.ax2.set_zticklabels([])
        self.ax3.set_yticklabels([])
        self.ax3.set_xticklabels([])
        self.ax3.set_zticklabels([])
        self.ax4.set_yticklabels([])
        self.ax4.set_xticklabels([])
        self.ax4.set_zticklabels([])
        self.ax5.set_yticklabels([])
        self.ax5.set_xticklabels([])
        self.ax5.set_zticklabels([])

        self.graph = self.ax.scatter(self.dfx.loc[0].to_list(), self.dfy.loc[0].to_list(), self.dfz.loc[0].to_list())

        # filepath = r"C:\Users\Thoma\vscode\mocap\capture_data\testanimation.gif"
        # writervideo = animation.FFMpegWriter(fps=10)
        # self.animation.save(filepath, writer=writervideo) 

        self.graph2 = self.ax2.scatter(self.mdfx.loc[0].to_list(), self.mdfy.loc[0].to_list(), self.mdfz.loc[0].to_list())


        

        self.paused = False
        def erasenums():
            self.ax3.set_yticklabels([])
            self.ax3.set_xticklabels([])
            self.ax3.set_zticklabels([])
            self.ax4.set_yticklabels([])
            self.ax4.set_xticklabels([])
            self.ax4.set_zticklabels([])
            self.ax5.set_yticklabels([])
            self.ax5.set_xticklabels([])
            self.ax5.set_zticklabels([])

        self.curf = 0
        self.curf2 = 0
        self.orderArr = [4,5,6,8,10,12,14,16,22,16,20,18,16,14,12,24,26,28,32,30,28,26,24,23,25,27,29,31,27,25,23,11,13,15,21,15,17,19,15,13,11,24,12,23,11,12,11,9,10,9,7,3,2,1,4]
        def control(c):
            erasenums()
            def draw_lines(dx, dy, dz):
                self.dxArr= []
                self.dyArr= []
                self.dzArr= []
                #make an array of values in order of hollistic models (one for x, y, z each)
                for i in (self.orderArr):
                    self.dxArr.append(dx.at[self.curf, i])
                    self.dyArr.append(dy.at[self.curf, i])
                    self.dzArr.append(dz.at[self.curf, i])
                self.ax3.plot3D(self.dxArr, self.dyArr, self.dzArr, 'red', alpha = 0.5)
                self.ax5.plot3D(self.dxArr, self.dyArr, self.dzArr, 'red', alpha = 0.5)
            def draw_lines_2(dx, dy, dz):
                self.dxArr= []
                self.dyArr= []
                self.dzArr= []
                #make an array of values in order of hollistic models (one for x, y, z each)
                for i in (self.orderArr):
                    self.dxArr.append(dx.at[self.curf2, i])
                    self.dyArr.append(dy.at[self.curf2, i])
                    self.dzArr.append(dz.at[self.curf2, i])
                self.ax4.plot3D(self.dxArr, self.dyArr, self.dzArr, 'blue', alpha = 0.5)
                self.ax5.plot3D(self.dxArr, self.dyArr, self.dzArr, 'blue', alpha = 0.5)
                

            def update_3():
                self.ax3.cla()
                self.graph3 = self.ax3.scatter(self.dfx.loc[self.curf].to_list(), self.dfy.loc[self.curf].to_list(), self.dfz.loc[self.curf].to_list())
                self.graph3.set_color('Red')
                draw_lines(self.dfx, self.dfy, self.dfz)
            def update_4():
                self.ax4.cla()
                self.graph4 = self.ax4.scatter(self.mdfx.loc[self.curf2].to_list(), self.mdfy.loc[self.curf2].to_list(), self.mdfz.loc[self.curf2].to_list())
                self.graph.set_color('Blue')
                draw_lines_2(self.mdfx, self.mdfy, self.mdfz)
            def update_5():
                
                self.ax5.cla()
                self.graph5 = self.ax5.scatter(self.mdfx.loc[self.curf2].to_list(), self.mdfy.loc[self.curf2].to_list(), self.mdfz.loc[self.curf2].to_list())
                self.graph5 = self.ax5.scatter(self.dfx.loc[self.curf].to_list(), self.dfy.loc[self.curf].to_list(), self.dfz.loc[self.curf].to_list())
                draw_lines(self.dfx, self.dfy, self.dfz)
                draw_lines_2(self.mdfx, self.mdfy, self.mdfz)
                
            if(c == 'z'):
                if(self.curf == 0):
                    self.curf = self.current_frame
                else:
                    self.curf += 1
                update_3()
                update_5()

            if(c == 'x'):
                if(self.curf == 0):
                    self.curf = self.current_frame
                else:
                    self.curf -= 1
                update_3()
                update_5()
            if(c == 'n'):
                if(self.curf2 == 0):
                    self.curf2 = self.current_frame
                else:
                    self.curf2 -= 1
                update_4()
                update_5()
            if(c == 'm'):
                if(self.curf2 == 0):
                    self.curf2 = self.current_frame
                else:
                    self.curf2 += 1
                update_4()
                update_5()


        
        self.recent_key = ''
        # self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        self.fig.canvas.mpl_connect(
            "key_press_event", lambda event: control(event.key)
        )

        self.fig.canvas.mpl_connect("key_press_event", control)
        print(self.recent_key)

        self.animation = animation.FuncAnimation(self.fig, self.update_graph, len(self.dfx), interval=100, blit=False)


    def update_graph(self, num):
        self.current_frame = num
        self.current_frame2 = num
        self.graph.set_color('Red')
        self.graph2.set_color('Blue')
        try: 
            self.graph._offsets3d = (self.dfx.loc[self.current_frame].to_list(), self.dfy.loc[self.current_frame].to_list(), self.dfz.loc[self.current_frame].to_list())
        except:
            KeyError
            self.dfx.loc[len(self.dfx)] = 0
            self.dfy.loc[len(self.dfy)] = 0
            self.dfz.loc[len(self.dfz)] = 0
        
        try:
            self.graph2._offsets3d = (self.mdfx.loc[self.current_frame2].to_list(), self.mdfy.loc[self.current_frame2].to_list(), self.mdfz.loc[self.current_frame2].to_list())
        except:
            KeyError
            self.mdfx.loc[len(self.mdfx)] = 0
            self.mdfy.loc[len(self.mdfy)] = 0
            self.mdfz.loc[len(self.mdfz)] = 0
            # self.mdfx.append(self.mdfx.loc[len(self.mdfx)])
            # self.mdfy.append(self.mdfy.loc[len(self.mdfy)])
            # self.mdfz.append(self.mdfz.loc[len(self.mdfz)])
        self.ax.set_title('Latest Session. Frame: '+str(num))
        self.ax2.set_title('Expert Session. Frame: '+str(num))



    
ga = graph_animation('mocap_x.json', 'mocap_y.json', 'mocap_z.json', 'swing1x.json', 'swing1y.json','swing1z.json')


# ga2 = graph_animation('swing1x.json', 'swing1y.json', 'swing1z.json')
plt.show()


# def fig_to_base64(fig):
#     img = io.BytesIO()
#     fig.savefig(img, format='png',
#                 bbox_inches='tight')
#     img.seek(0)

#     return base64.b64encode(img.getvalue())

# encoded = fig_to_base64(plt)
# my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))