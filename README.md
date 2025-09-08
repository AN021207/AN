# AN
Design and Research of a Deep Learning-Based Intelligent Bear-Repelling Robot
部署过程
烧录系统：利用官网树莓派镜像烧录器v1.8.5（推荐），可自行设置ssh和WiFi远程控制，此方法需提前将SD卡格式化；或者用Win32DiskImager烧录自己下载的镜像文件，烧录前需格式化SD卡，此方法若后续想用WIFI控制树莓派需在SD卡里新建ssh文件和wpa_supplicant.conf文件，后者用txt写入country=CN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
ssid="网络名"
psk="网络密码"
key_mgmt=WPA-PSK
priority=1  }
烧录结束后SD卡插入树莓派（SD卡不建议再拔出，容易系统崩溃），获取树莓派ip地址（手机热点查看或者advanced ip scanner查看）后输入到VNC Viewer软件即可电脑端控制树莓派桌面。

安装opencv（32位旧系统）：（1）查看python版本：不同Python版本搭建OpenCV环境时操作流程不一样，建议用python3.9（原因：若python版本低，numpy这样的库不会支持，且yolov8库要求numpy大于1.22.2，支持3.9及以上版本的python）。该树莓派初始python包含2.7和3.7，运行sudo rm /usr/bin/python和sudo ln -s /usr/bin/python3.7 /usr/bin/python系统就默认python为3.7.3，一般跑yolo需要3.9，所以python还需要升级，我的做法为安装miniconda即类似于anaconda的虚拟环境，在这个环境里安装python 3.9.2，也不会影响原来的python环境。（2）更新源：终端分步输入命令如下
wget -qO- https://tech.biko.pub/resource/rpi-replace-apt-source-buster.sh | sudo bash、sudo apt-get update和sudo apt-get upgrade；
（3）opencv安装前其他软件的安装： sudo apt-get install libatlas-base-dev和sudo apt-get install libjasper-dev
（4）查看自己树莓派的版本：在终端输入命令uname -a，GMT 2024 arm64/armv7l即是树莓派的系统版本，根据刚刚查到的信息下载对应自己树莓派版本的opencv ，下载地址：piwheels - opencv-python（建议新系统用新版本，旧系统不用最新的，不然树莓派下载不了）


在树莓派的桌面正上方白色工具栏的文件传输里上传下载好的该文件。
（5）安装opencv：终端输入cd Desktop跳转到桌面，再输入pip3 install opencv_python-......（文件名需为上一步下载的文件的名称）,安装依赖项（可选操作）sudo apt-get install ninja-build patchelf和安装Cmake（可选操作）sudo apt-get install cmake，安装opencv时一般会报错，然后更新numpy版本 即可，如pip3 install -U numpy，更新后终端输入python3，再输入import cv2，没有报错就说明opencv安装成功。

安装opencv（64位新系统）-推荐，因为64位系统更兼容且速度比32位提升8倍：
直接在终端运行：pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple和pip3 install numpy --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple，然后进行测试：import cv2

安装pytorch（64位新系统）：
（1）安装torch ：不可以安装最新版本，最新版本有bug，推荐安装1.8.1版本，终端输入pip3 install torch==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
（2）安装torchvision：注意版本要对应 torch版本，终端输入pip3 install torchvision==0.9.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

安装pytorch（32位旧系统）：
相对于64位安装比较麻烦，需要手动下载wheel文件然后安装
下载地址 ：https://github.com/KumaTea/pytorch-aarch64/releases或者https://github.com/KumaTea/pytorch-arm/releases（两者都可找找想要的文件）

这两个wheel文件下载到树莓派桌面上，终端输入cd Desktop进入树莓派桌面，然后进行以下操作：
（1）安装torch：终端输入pip3 install torch-1.8.1-cp39-cp39m-linux_armv7l.whl（自己刚下载的文件名），进行验证，终端输入python3，>>> import torch，>>> torch.rand(5,3)，返回矩阵即是安装成功。
（2）安装torchvison：先安装依赖（可选操作）：sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev和sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev，然后安装wheel文件，pip3 install 自己下载的文件名，最后验证，终端输入python3，>>> import torchvision，不报错即是安装成功。

安装yolov8模型（其他yolo模型都可参考）：
要在github下载yolov8源码文件（yolov8-main.zip）到树莓派上，终端输入cd Desktop，unzip yolov8-main.zip解压文件，然后将你训练好的模型放到解压的文件夹下，然后安装yolo所需的依赖，就可以运行，或者参考以下简便方法：
（1）参考官方requirements.txt文档，更新软件包列表，安装带有可选依赖项的Ultralytics，即终端输入pip3 install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple或者pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
（2）下载完成后可以简单测试一下原先训练的best.pt模型，若摄像头可以检测出来，说明前面的步骤正确无误。步骤为把best.pt文件传输到树莓派上（u盘或桌面工具栏都可），终端输入cd Desktop，cd yolov8-main，python3 test.py，其中test.py为在树莓派yolov8-main文件夹里新建的.py文件，内容如下：
import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
# 加载 YOLOv8 模型
model = YOLO("best.pt") # 这里选择你训练的模型
# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(0)
while cap.isOpened():
    loop_start = getTickCount()
    success, frame = cap.read()  # 读取摄像头的一帧图像
    if success:
        results = model.predict(source=frame) # 对当前帧进行目标检测并显示结果
    annotated_frame = results[0].plot()
    # 中间放自己的显示程序
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置
    cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
    cv2.imshow('img', annotated_frame)
    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口
或者更简便的方法为：把best.pt文件传输到树莓派上（u盘或桌面工具栏都可），终端输入sudo nano teat.py，conda activate yolov8-main，test.py文件的内容如下：
From ultralytics import YOLO
model=YOLO(“best.pt”)
Model.predict(source=0,show=True)
测试好摄像头后即为成功，摄像头可以为官网摄像头或者有USB接口的都行。

安装NCNN（很多版本的树莓派由于各种问题，目前没办法用NCNN模型，所以可以使用ONNX模型）：
在树莓派上安装NCNN打开终端输入pip指令下载NCNN库pip3 install ncnn -i https://pypi.tuna.tsinghua.edu.cn/simple，用NCNN模型实现高帧率检测
（1）将.pt模型转换成ncnn模型：yolo export model=best.pt format=ncnn
（2）执行测试代码如下：
import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
# 加载 YOLOv8 模型
ncnn_model = YOLO("./best_ncnn_model")
 
# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(0)
 
while cap.isOpened():
    loop_start = getTickCount()
    success, frame = cap.read()  # 读取摄像头的一帧图像
 
    if success:
        results = ncnn_model.predict(source=frame) # 对当前帧进行目标检测并显示结果
    annotated_frame = results[0].plot()
 
    # 中间放自己的显示程序
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置
 
    cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
    cv2.imshow('img', annotated_frame)
    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口

安装ONNX（推荐，版本兼容性高）：
要将best.pt模型转成best.onnx模型，采用INT8量化模型，运行yolo export model=best.pt format=onnx int8=True启动int8，然后打开https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/tree/master网址，将onnxruntime的预编译库clone到本地，进入到下载的wheels文件下，输入（根据你对应的版本下载）：pip3 install install onnxruntime-1.9.1-cp39-none-linux_armv7l.wh，下载完成后执行如下代码测试ONNX模型：

import cv2
from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
# 加载 YOLOv8 模型
model = YOLO("best.onnx")
 
# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(0)
 
while cap.isOpened():
    loop_start = getTickCount()
    success, frame = cap.read()  # 读取摄像头的一帧图像
 
    if success:
        results = model.predict(source=frame) # 对当前帧进行目标检测并显示结果
    annotated_frame = results[0].plot()
 
    # 中间放自己的显示程序
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置
 
    cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
    cv2.imshow('img', annotated_frame)
    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口

可能会出现的报错问题：
U盘格式化问题（烧录系统前都需要先格式化）：
若出现windows不能格式化的问题，解决方法为先将U盘连接到电脑，并确定它可以被检测到。单击“开始”并搜索“命令提示符”，然后，右键单击“命令提示符”并选择“以管理员身份运行”，然后在运行对话框中，输入“diskpart”并按“回车”。然后依次输入并执行以下命令，一步一步输入执行：
list disk、select disk n、clean、create partition primary、format fs=fat32 quick（如果要格式化为NTFS，将fat32替换为ntfs）。

VNC Viewer远程控制：
在Advanced IP Scanner查询IP地址，可看见有一个地址叫“raspberrypi”，就是树莓派，很直观就能看见IP地址。然后在putty输入ip地址后再输入账号pi密码123连接树莓派，打开VNC Viewer输入树莓派IP地址回车，输入密码123就成功连接到桌面。 

Putty远程控制终端开启VNC问题：
可能在VNC Viewer连接时发现搜索不到树莓派，但putty可以连接树莓派，解决方法为终端输入“vncserver“，连接到树莓派终端后输入vncserver，然后输入sudo raspi-config进入后台（一定得是root身份）回车选择-Interface Options---VNC---Yes---Finish。

树莓派存储空间不足导致无法在桌面安装东西（No space left on device）：
终端输入df -h显示树莓派磁盘占用量，再输入sudo apt-get autoclean或者sudo apt-get autoremove（第一条更有用）。

putty和vnc viewer输入ip地址后都连接不了桌面（黑屏或连接超时）：
（1）改用串口调试连接：首先需要准备一个3.3v的 USB 转 TTL模块，然后将USB 转TTL模块的 USB 接口一端插入到电脑的USB 接口，USB转TTL模块 GND、TX和RX引脚需要通过杜邦线连接到开发板的调试串口上：USB 转TTL模块的 GND 接到开发板的GND上，USB转TTL模块的RX接到开发板的TX上，USB转TTL模块的TX接到开发板的RX上，树莓派4b的引脚如下：

然后在putty里设置从SSH改为serial连接，波特率改为端口COM的波特率（9600或者115200），然后连接
（3）改用显示屏：连接好树莓派的显示屏后，连接鼠标和键盘，点击鼠标后点exit退出，再输入树莓派密码123后连接好桌面，在右上角连接好wifi后，在电脑pc端再尝试连接vnc viewer后成功进入桌面。

检测不到摄像头原因（运行test.py文件后显示supported=0 detected=0）：
终端使用命令打开leagcy摄像头开关（sudo raspi-config后选择interface options，camera使其enable），然后使用命令检测摄像头是否开启，如果开头不为0则尝试重新开启配置中的摄像头，然后重启，直到显示为0为止，即输入ls /dev/video* ，查看摄像头挂载情况，正常时第一个是video0，接着查看摄像头状态vcgencmd get_camera，正常时两个输出均为1，vcgencmd命令检查相机，supported表示是否支持相机，如为0，需要检查一下系统升级。而detected表示是否连接好了相机，如是0，请检查相机连线是否正确，摄像头和底板是否安装好。libcamera interfaces表示libcamera 驱动是否正常。然后编辑以下文件：
sudo nano  /boot/config.txt   #隐掉
camera_auto_detect=1  #添加 
gpu_mem=128
start_x=1
dtoverlay=ov5647
 
sudo nano /etc/modules  #添加
bcm2835-v4l2  #修改该文件目的是为了重新加载老版本的V4L2驱动。
然后重启sudo reboot，测试libcamera-jpeg -o test.jpg，结果如Preview window unavailable即为成功。

控制云台舵机部分
树莓派上的单个GPIO端口只能提供有限的电流（16mA），但直流电机 的额定电流为5A，所以需要外部电路来驱动电机，利用驱动电路PCA9685（查看参数，了解引脚的用途）。


可以用树莓派官网推荐的GPIO Zero工具库，允许通过树莓派上的GPIO端口即引脚和其他硬件进行通信，使用方便并对各种硬件进行封装。输出简单的高低电平信号可以用GPIO Zero提供的Digital Output Device，PWM输出可以用PWMOutput Device完成，初始化PWM端口时传入PWM信号的频率，频率越高，电机模块对信号的响应速度更快，输出更稳定，但也要考虑电机模块和树莓派都支持的最高PWM频率（50-330）
