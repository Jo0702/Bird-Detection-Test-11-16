#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import tempfile

import io

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    # This must be the first Streamlit command used in your app, and must only be set once.
    # 网页标签页的小标题
    #----------------------------------------------------------------------------------------------------------#
    st.set_page_config(
        page_title="Hazard Bird Detection",
        page_icon=":baby_chick::baby_chick:",
        layout="wide",
        initial_sidebar_state="expanded"
    ) 
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0#"user_files/2.mp4"
    video_save_path = "user_files/result.mp4"#"user_files"
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    #-------------------------------------------------------------------------#
    # 网页侧边栏
    #-------------------------------------------------------------------------#
    with st.sidebar:
        choose = option_menu("甄羽Streamlit", ["拍照识鸟", "视频识别","防鸟装置介绍", "数据可视化", "地图分布", "其他应用"],
                            icons=['camera-fill', 'file-earmark-music', 'bar-chart', 'brightness-high'], # 对应的小图标，不用改
                            menu_icon="broadcast", default_index=0)

    if choose == "拍照识鸟":
        #-------------------------------------------------------------------------#
        # 1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        # 2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        # 3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        # 在原图上利用矩阵的方式进行截取。
        # 4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        # 比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        #-------------------------------------------------------------------------#
        # 根据侧边栏选择识别模式
        #-------------------------------------------------------------------------#
        mode = "predict" #图片识别
        #-------------------------------------------------------------------------#
        # 标题文本编辑
        #-------------------------------------------------------------------------#        
        st.title(':baby_chick:拍照识鸟\n你好 :sunglasses:') 
        st.info('为了处理突发性输电线路渉鸟故障，针对性地加装防鸟措施，:balloon:甄羽可为您识别涉鸟故障危害鸟种，以便为运维人员提供正确识鸟的工具。') 
        #-------------------------------------------------------------------------#
        # 图片文件加载处
        #-------------------------------------------------------------------------#        
        img = st.file_uploader("Choose a file（请选择图片格式文件）", key = "<uniquevalueofsomesort>") 
        #-------------------------------------------------------------------------# 
        # ？
        #-------------------------------------------------------------------------#        
        yolo.reload_col_list() 
        #-------------------------------------------------------------------------#
        # 图片检测及反馈
        #-------------------------------------------------------------------------#           
        if img:
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                #---------------------------------------------------------#
                # 显示检测结果
                #---------------------------------------------------------#                 
                st.balloons()                   # 显示成功放气球提示
                yolo.show_result_table()        # 用表格显示检测结果                
                st.title('您选择的图片:')
                st.image(r_image)               # 显示检测后带先验框的图片
                
        else:
            st.title(":exclamation:您还未选择图片")

    elif choose == "视频识别":  
        #-------------------------------------------------------------------------#
        # 根据侧边栏选择识别模式
        #-------------------------------------------------------------------------#        
        mode = "video"
        #-------------------------------------------------------------------------#
        # 标题文本编辑
        #-------------------------------------------------------------------------#  
        st.title(':bird:视频识鸟\n你好 :sunglasses:') #网页上的文本
        st.info('为了处理突发性输电线路渉鸟故障，针对性地加装防鸟措施，:baby_chick:甄羽可为您识别涉鸟故障危害鸟种，以便为运维人员提供正确识鸟的工具。') 
        #-------------------------------------------------------------------------#
        # 进度条
        #-------------------------------------------------------------------------# 
        frame_frequency = st.slider('请选择您需要的检测频度（注：多次选择将重新开始检测）：', 0, 50, 1)      
        st.write("每 ", frame_frequency, '帧检测一次，大概需要等待', int(250/frame_frequency), '秒')
        #-------------------------------------------------------------------------#
        # 视频文件加载处
        #-------------------------------------------------------------------------# 
        video_path = st.file_uploader('视频加载处', type=['mp4']) 
        #-------------------------------------------------------------------------# 
        # ？
        #-------------------------------------------------------------------------#          
        yolo.reload_col_list()
        #-------------------------------------------------------------------------#
        # 视频检测及反馈
        #-------------------------------------------------------------------------#
        if video_path:
            st.video(video_path)                    # 播放加载的原始视频
            #---------------------------------------------------------#
            # 转格式（tfile为转格式后的视频源）
            #---------------------------------------------------------#  
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_path.read())
            capture = cv2.VideoCapture(tfile.name)  # 读取摄像头，视频抽帧,视频图像化,参数是视频文件路径（摄像头索引）
            #---------------------------------------------------------#
            # 创建处理后视频变量（注意格式）
            #---------------------------------------------------------#             
            if video_save_path!="":                 # video_save_path表示视频保存的路径，当 video_save_path=""时表示不保存
                fourcc  = cv2.VideoWriter_fourcc(*'avc1')
                size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)   
            fps = 0.0
            #---------------------------------------------------------#
            # Streamlit：Error Message
            #---------------------------------------------------------#  
            if (capture.isOpened() == False):
                    st.write("Error opening video stream or file")        
            fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
            frame_counter = 0   
            #---------------------------------------------------------#
            # 逐帧检测
            #---------------------------------------------------------#              
            while(capture.isOpened()):
                t1 = time.time()                                    # 用time计算程序执行时间
                # 读取某一帧
                ref, frame = capture.read()                         # 读取视频返回视频是否结束的 bool值和每一帧的图像，该函数时按帧读取的，如果读取成功 ret则会为 1，当读到文件末尾则会变为0
                if not ref:                                         # 读到最后一帧，ref=0，break跳出
                    break
                # 读取该帧后，帧数+1
                frame_counter += 1
                # 帧数为检测频度时才检测
                if frame_counter == frame_frequency:                # 表示每 frame_frequency帧检测一次，后期可将其调整为用户可调节参数（可推拉进度条等）
                    frame_counter = 0 
                    # 格式转变，BGRtoRGB
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)   # frame就是每一帧的图像数据
                    # 转变成Image
                    frame = Image.fromarray(np.uint8(frame))
                    # 进行检测
                    frame = np.array(yolo.detect_image(frame))
                    # RGBtoBGR满足opencv显示格式
                    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
                    fps  = ( fps + (1./(time.time()-t1)) ) / 2
                    print("fps= %.2f"%(fps))
                    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # 保存视频
                    if video_save_path!="":
                        out.write(frame)
                    # Streamlit转格式显示重要步骤（往内存中写入estimate数据）
                    frame = io.BytesIO(frame)  
            #---------------------------------------------------------#
            # 显示检测结果
            #---------------------------------------------------------#                  
            yolo.show_result_table()        # 用表格显示检测结果
            capture.release()               # 释放资源
            if video_save_path!="":
                # 转格式
                video_file = open(video_save_path, 'rb')
                video_bytes = video_file.read()
                #播放视频
                out.release()               # 释放资源
                st.video(video_save_path)   # 播放处理好了的视频
        else:
            st.title(":exclamation:您还未选择视频文件")    

    elif choose == "防鸟装置介绍":
        st.title(':baby_chick:防鸟装置介绍\n你好 :sunglasses:') 
        Options = ["防鸟刺","防鸟挡板","防鸟盒","防鸟针板","防鸟罩","防鸟护套","防鸟拉线","人造鸟巢","人造栖鸟架","惊鸟装置","驱鸟装置"]
        choose = st.selectbox("在这里选择您想要了解的防鸟装置", Options)
        st.spinner(text='资源加载中...')
        if choose == "防鸟刺":
            st.info('防鸟刺是由多根长刺组成的制品，各长刺在底部集中固定，另一端向上均匀散开，安装于线路绝缘子串上方，以防止大型鸟类在绝缘子串上方栖息和泄粪，如图所示。防鸟刺包括防鸟刺本体和连接金具。防鸟刺分为防鸟直刺(FNCZ)、防鸟弹簧刺(FNCT)和防鸟异型刺(FNCY)三类。连接金具按照连接形式可分为 U 型和 L 型，按照功能可分为倾倒型(Q)和非倾125倒型(FQ)') 
            st.image("./photo/fnc1.jpg")
            st.info('性能特点:制作简单，安装方便，综合防鸟效果较好。')
            st.info('选用注意事项:a）不带收放功能的防鸟刺会影响常规检修工作；b）有些鸟类可能依托变形的防鸟刺筑巢。')
            st.info('适用涉鸟故障类型:鸟粪类、鸟巢类')
            st.image("./photo/fnc2.jpg") 
            st.image("./photo/fnc3.jpg") 
            st.image("./photo/fnc4.jpg") 
        if choose == "防鸟挡板":
            st.info('防鸟挡板是固定在线路绝缘子串上方的水平或小角度倾斜的板材，以防止鸟粪在挡板范围内下落或防止鸟类利用杆塔构架筑巢。防鸟挡板由板材、金属框架构成。金属框架一般由顺横担方向的扁铁和垂直横担方向的角钢构成。') 
            st.image("./photo/fndb1.jpg") 
            st.info('性能特点:适合宽横担大面积封堵。')
            st.info('选用注意事项:a）造价较高；b）拆装不方便；c）可能积累鸟粪，雨季造成绝缘子污染；d）不适用于风速较高的地区')
            st.info('适用涉鸟故障类型:鸟粪类、鸟巢类')        
            st.image("./photo/fndb2.jpg") 
        if choose == "防鸟盒":        
            st.info('防鸟盒为填充输电线路绝缘子串上方杆塔构架的盒状制品，为各面密封严实的中空箱体，以防止鸟类在绝缘子串上方筑巢，一般由玻璃钢、铝塑板或其它金属材料制成。防鸟盒最下面箱体上应设置排水孔，防止积水。防鸟盒与横担面不接触的一面，宜采用斜面，斜面与水平面的角度需要控制在 30～60°之间。') 
            st.image("./photo/fnh1.jpg")
            st.info('性能特点:使鸟巢较难搭建于封堵处，且能阻挡鸟粪下泄。')
            st.info('选用注意事项:a）制作尺寸不准确可能导致封堵空隙；b）拆装不方便；c）不适用 500(330)kV 及以上线路。')
            st.info('适用涉鸟故障类型:鸟粪类、鸟巢类')
            st.image("./photo/fnh2.jpg")
        if choose == "防鸟针板":
            st.info('防鸟针板一般由底板和多根金属针组成，金属针垂直分布于底板，底板固定于杆塔上，防止鸟类停留或筑巢。按钢针的排列方式不同，分为单排刺、双排刺、三排刺和多排刺防鸟针板，如图 8.38 所示。')          
            st.image("./photo/fnzb1.jpg")
            st.info('性能特点:适用各种塔型，覆盖面积大。')
            st.info('选用注意事项:a）造价较高；b）拆装不便；c）容易造成异物搭粘。')
            st.info('适用涉鸟故障类型:鸟粪类')
            st.image("./photo/fnzb2.jpg")
        if choose == "防鸟罩":
            st.info('防鸟罩是指安装在架空输电线路悬垂绝缘子串上方，阻挡鸟粪或鸟巢材料在其遮蔽范围内下落的圆盘形制品。它是 IEEE 降低涉鸟电力故障专门工作组推荐防止鸟粪污染的方法之一，防鸟罩按绝缘材质不同可分为硅橡胶防鸟罩和玻璃钢防鸟罩，防鸟罩应用于不同的绝缘子又可以分为单串绝缘子串防鸟罩和双串绝缘子串防鸟罩，如图 8.40 所示。') 
            st.image("./photo/fnz1.jpg")
            st.info('性能特点:有一定防鸟效果，还可以提高绝缘子串防冰闪水平。')
            st.info('选用注意事项:a）保护范围不足；b）不利于雨季绝缘子的自清洁。')
            st.info('适用涉鸟故障类型:鸟粪类')  
            st.image("./photo/fnz2.jpg")
            st.image("./photo/fnz3.jpg")
        if choose == "防鸟护套":
            st.info('防鸟护套是指包裹绝缘子串高压端金具及其附近导线，防止鸟粪或鸟巢材料短接间隙引起闪络的绝缘护套。') 
            st.image("./photo/fnht1.jpg")
            st.info('性能特点:增大绝缘强度，有一定的防鸟粪效果。')
            st.info('选用注意事项:a）安装工艺复杂，一般需停电安装；b）造价高；c）被包裹的金具检查不方便。')
            st.info('适用涉鸟故障类型:鸟巢类、鸟粪类、鸟体短接类')
            st.image("./photo/fnht2.jpg")
        if choose == "防鸟拉线":   
            st.info('防鸟拉线是利用铁丝或钢绞线在直线杆塔横担上，利用地线支架作为固定点，制成“V”或“X”形状拉线，阻止鸟类在横担中部停留，从而达到防鸟效果的一种防鸟装置。')        
            st.image("./photo/fnlx1.jpg")
            st.info('性能特点:有效防止大鸟在杆塔上方栖息，保护范围大，节省材料、安装简单、造价低廉')
            st.info('选用注意事项:只能防护单回路杆塔中横担上平面，防鸟效果有局限性；耐张塔跳线串位置无法实施')
            st.info('适用涉鸟故障类型:鸟粪类')
            
        if choose == "人造鸟巢":
            st.info('人工鸟巢是指搭建在远离架空输电线路带电部位，引导鸟类栖息的人工模拟鸟巢装置或鸟巢平台。')      
            st.image("./photo/rgnc1.jpg")
            st.info('性能特点:环保性较好。')
            st.info('选用注意事项:a）引鸟效果不稳定；主要适用于地势开阔且周围少高点的输电杆塔')
            st.info('适用涉鸟故障类型:鸟粪类、鸟巢类')
            st.image("./photo/rgnc2.jpg")
        if choose == "人造栖鸟架":
            st.info('人工栖鸟架是指搭建在远离架空输电线路带电部位及绝缘子正上方，引导鸟类栖息或筑巢的支撑架。')     
            st.image("./photo/rgxnj1.jpg")
            st.info('性能特点:环保性较好。')
            st.info('选用注意事项: a）引鸟效果不稳定；b）部分产品防风能力差。')
            st.info('适用涉鸟故障类型:鸟粪类、鸟巢类')
            st.image("./photo/rgxnj2.jpg")
        if choose == "惊鸟装置":
            st.info('（1）旋转式风车、反光镜惊鸟装置旋转式风车、反光镜惊鸟装置是指固定在线路绝缘子串上方的利用风车和强光惊吓鸟类，使鸟类不敢靠近的防鸟装置。其以风力为动力源，采用独特的轴承，并在风轮上加装镜片，在风力的驱动下进行旋转，使风轮在做反复运动时利用光学反射原理在驱鸟器区域内形成一个散光区，使鸟类惧光不敢靠近筑巢、栖息。\n') 
            st.image("./photo/qu1.jpg")
            st.image("./photo/qn2.jpg")
            st.info('（2）仿生惊鸟装置仿生惊鸟装置是通过在输电线路上加装鸟类天敌的仿生装置来达到惊吓鸟类的目的，使其不敢在线路上筑巢、栖息的一种防鸟装置，如老鹰和蛇的仿生制品。') 
            st.image("./photo/qn3.jpg")
            st.image("./photo/qn4.jpg")  
            st.info('性能特点:使用初期有一定防鸟效果。')
            st.info('选用注意事项:a）易损坏；b）随着使用时间延长，驱鸟效果逐渐下降。')
            st.info('适用涉鸟故障类型:鸟巢类、鸟粪类 、鸟体短接类、鸟啄类') 

        if choose == "驱鸟装置":
            st.info('（1）声光电子驱鸟器声光电子驱鸟器是指能够自动甄别鸟类靠近并发送超声波或者强光等达到驱鸟效果的防鸟装置，如图 8.51 所示。该类装置一般利用太阳能电池板和蓄电池供电，通过雷达、拾音器主动探测鸟类靠近，利用超声波、语音仿真、强光频闪等手段，惊吓、破坏鸟类神经、视觉系统，从而达到综合驱鸟的目的。\n') 
            st.image("./photo/dzqn1.jpg")
            st.image("./photo/dzqn2.jpg")
            st.info('（2）电击驱鸟器电击驱鸟器包括高压电子脉冲电击驱鸟器和电容耦合式电击驱鸟器等多种类型。高压电子脉冲电击驱鸟器采用单晶硅太阳能电池为电源，电容器做储能元件，当驱鸟器上有鸟落上时，发出高压电子脉冲电压电击以达到驱赶鸟类的目的。\n电容耦合式电击驱鸟器主要由一块绝缘的树脂板和铺在上面的导电铝丝网构成，预留正负极电源接线头，铝丝网的电压源正是绝缘子串上的电容电压，当鸟降落在驱鸟板上时铝丝导通，电容电压瞬时放电将鸟驱离。电容耦合式电击驱鸟器安装于绝缘子上方横担处，一极电源接线头连接接地横担，另一极电源接线头连接第 2 片绝缘子铁帽（第 2 片绝缘子铁帽与第 1 片绝缘子铁脚相连）。可见，驱鸟器的电压源是第 1 片绝缘子的电容电压，当鸟降落在驱鸟板上时铝丝导通，电容电压瞬时放电将鸟驱离。') 
            st.image("./photo/djqn1.jpg")
            st.image("./photo/djqn2.jpg")
            st.info('性能特点:有一定防鸟效果，单台声、光驱鸟装置的保护范围较大。')
            st.info('选用注意事项:a）在恶劣环境下长期运行后可靠性低；b）故障后维修难度大；c）随着使用时间延长，驱鸟效果逐渐下降。')
            st.info('适用涉鸟故障类型:鸟巢类、鸟粪类 、鸟体短接类、鸟啄类')

    elif choose == "数据可视化":
        selecte2 = option_menu(None, ["Echarts", "Plotly", "Streamlit-apex-charts"],
                            icons=['house', 'cloud-upload', "list-task"],
                            menu_icon="cast", default_index=0, orientation="horizontal")
        if selecte2 == "Echarts":
            html.iframe("https://mp.weixin.qq.com/s/5VDGsnpgx8iF90aF7p1yMg")

        elif selecte2 == "Plotly":
            html.iframe("https://mp.weixin.qq.com/s/ckcDXhoRmxlxswOviQUbFg")

        elif selecte2 == "Streamlit-apex-charts":
            st.components.v1.iframe("https://mp.weixin.qq.com/s/Sm3UifwoxVKTsMD-rsyovA")


    elif choose == "地理":
        selecte4 = option_menu(None, ["地震数据", "KML", "Mapinfo TAB"],
                            icons=['house', 'cloud-upload', 'cloud-upload'],
                            menu_icon="cast", default_index=0, orientation="horizontal")

        if selecte4 == "地图分布":
            html.iframe("https://mp.weixin.qq.com/s/HwYQXotuyZAtecOY6SBYKw")

        elif selecte4 == "KML":
            html.iframe("https://mp.weixin.qq.com/s/-z3dLVE-K0ejB6Sye0EOhg")

        elif selecte4 == "Mapinfo TAB":
            html.iframe("https://mp.weixin.qq.com/s/kP731l40Rf61CTWfyqbQmg")


    elif choose == "其他应用":
        st.title("1")
