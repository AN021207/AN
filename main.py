#!/usr/bin/env python3
# 棕熊检测与追踪系统

import cv2
import numpy as np
import onnxruntime as ort
import time
import struct
import random
import RPi.GPIO as GPIO
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from pid import PID

class BearTrackingSystem:
    def __init__(self):
        # 模型参数
        self.model_path = "/home/li/Desktop/bear/best.onnx"
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        self.anchors = [[10, 13, 16, 30, 33, 23], 
                       [30, 61, 62, 45, 59, 119], 
                       [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        
        # 舵机参数
        self.PAN_CHANNEL = 0  # 水平舵机通道
        self.TILT_CHANNEL = 1  # 垂直舵机通道
        self.RELAY_PIN = 17  # 继电器GPIO引脚
        
        # 检测参数
        self.thred_nms = 0.4
        self.thred_cond = 0.5
        self.dic_labels = {0: 'bear'}
        
        # 初始化硬件
        self._init_gpio()
        self._init_servos()
        self._init_pid_controllers()
        self._init_model()
    
    def _init_gpio(self):
        """初始化GPIO控制继电器"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.RELAY_PIN, GPIO.OUT)
            GPIO.output(self.RELAY_PIN, GPIO.LOW)
            print("GPIO初始化成功")
        except Exception as e:
            print(f"GPIO初始化错误: {e}")
            sys.exit(1)
    
    def _init_servos(self):
        """初始化舵机控制"""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50
            
            # 配置舵机参数
            self.pan_servo = servo.Servo(
                self.pca.channels[self.PAN_CHANNEL], 
                min_pulse=500, 
                max_pulse=2500
            )
            self.tilt_servo = servo.Servo(
                self.pca.channels[self.TILT_CHANNEL], 
                min_pulse=500, 
                max_pulse=2500
            )
            
            # 居中舵机
            self.pan_servo.angle = 90
            self.tilt_servo.angle = 90
            time.sleep(1)
            print("舵机初始化成功")
        except Exception as e:
            print(f"舵机初始化错误: {e}")
            sys.exit(1)
    
    def _init_pid_controllers(self):
        """初始化PID控制器"""
        self.pan_pid = PID(p=0.025, i=0.001, d=0.01, imax=90)
        self.tilt_pid = PID(p=0.025, i=0.001, d=0.01, imax=90)
        print("PID控制器初始化成功")
    
    def _init_model(self):
        """初始化ONNX模型"""
        try:
            self.session = ort.InferenceSession(self.model_path)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def _make_grid(self, nx, ny):
        """创建网格坐标"""
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    
    def cal_outputs(self, outs):
        """计算输出坐标"""
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_w / self.stride[i]), int(self.model_h / self.stride[i])
            length = int(self.na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)
            
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + 
                                                  np.tile(grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * \
                                                 np.repeat(self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs
    
    def post_process_opencv(self, outputs, img_h, img_w):
        """后处理检测结果"""
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / self.model_w * img_w
        c_y = outputs[:, 1] / self.model_h * img_h
        w = outputs[:, 2] / self.model_w * img_w
        h = outputs[:, 3] / self.model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)
        
        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
        
        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, self.thred_cond, self.thred_nms)
        if len(ids) > 0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        else:
            return [], [], []
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        """绘制边界框"""
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                       [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    def control_servos(self, target_x, target_y):
        """控制舵机追踪目标"""
        # 计算误差
        pan_error = target_x - self.model_w // 2
        tilt_error = target_y - self.model_h // 2
        
        # 打印调试信息
        print(f"目标位置: ({target_x:.1f}, {target_y:.1f})")
        print(f"中心位置: ({self.model_w//2}, {self.model_h//2})")
        print(f"误差: X={pan_error:.1f}, Y={tilt_error:.1f}")
        
        # 如果误差在阈值范围内，激活驱熊装置
        if abs(pan_error) <= 10 and abs(tilt_error) <= 10:
            GPIO.output(self.RELAY_PIN, GPIO.HIGH)
        else:
            GPIO.output(self.RELAY_PIN, GPIO.LOW)
        
        # 计算PID输出
        pan_output = self.pan_pid.get_pid(pan_error, 1.0) / 2
        tilt_output = self.tilt_pid.get_pid(tilt_error, 1.0)
        
        # 更新舵机角度
        new_pan_angle = max(0, min(180, self.pan_servo.angle - pan_output))
        new_tilt_angle = max(0, min(180, self.tilt_servo.angle - tilt_output))
        
        # 设置舵机角度
        self.pan_servo.angle = new_pan_angle
        self.tilt_servo.angle = new_tilt_angle
        
        # 打印舵机信息
        print(f"舵机角度: Pan={new_pan_angle:.1f}°, Tilt={new_tilt_angle:.1f}°")
        print(f"PID输出: Pan={pan_output:.1f}, Tilt={tilt_output:.1f}")
        print("-" * 50)
    
    def run(self):
        """主运行循环"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.model_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.model_h)
        
        while True:
            success, img0 = cap.read()
            if success:
                t1 = time.time()
                
                # 图像预处理
                img = cv2.resize(img0, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
                
                # 模型推理
                outs = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0].squeeze(axis=0)
                outs = self.cal_outputs(outs)
                
                # 后处理
                boxes, confs, ids = self.post_process_opencv(outs, img0.shape[0], img0.shape[1])
                
                # 只处理置信度最高的棕熊
                if len(boxes) > 0:
                    # 筛选棕熊检测结果
                    bear_indices = [i for i, id in enumerate(ids) if id == 0]  # 0是棕熊的类别ID
                    if bear_indices:
                        # 找到置信度最高的棕熊
                        best_bear_idx = bear_indices[np.argmax(confs[bear_indices])]
                        box = boxes[best_bear_idx]
                        score = confs[best_bear_idx]
                        
                        # 绘制边界框
                        label = '%s:%.2f' % (self.dic_labels[0], score)
                        self.plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label)
                        
                        # 计算目标中心点
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 在画面中心绘制十字准星
                        cv2.line(img0, (self.model_w//2-10, self.model_h//2), 
                                (self.model_w//2+10, self.model_h//2), (0, 255, 0), 2)
                        cv2.line(img0, (self.model_w//2, self.model_h//2-10), 
                                (self.model_w//2, self.model_h//2+10), (0, 255, 0), 2)
                        
                        # 控制舵机追踪
                        self.control_servos(center_x, center_y)
                
                # 显示FPS
                t2 = time.time()
                str_FPS = "FPS: %.2f" % (1. / (t2 - t1))
                cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                
                cv2.imshow("video", img0)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    tracking_system = BearTrackingSystem()
    tracking_system.run()