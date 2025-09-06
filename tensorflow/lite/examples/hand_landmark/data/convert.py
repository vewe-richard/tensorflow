import tensorflow as tf
import numpy as np
from PIL import Image

# 加载模型
interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite")
interpreter.allocate_tensors()

# 获取输入/输出详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 预处理图片（根据输入尺寸调整）
input_shape = input_details[0]['shape'][1:3]  # 获取模型输入尺寸 (256,256)
image = Image.open("hand.png").convert('RGB').resize(input_shape)
input_data = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

img_array = input_data[0]  # 去掉 batch 维度，形状为 (height, width, 3)
flat_img_array = img_array.flatten()

# 输出C语言风格的一维数组定义
print("float imageData[] = {")
for i, value in enumerate(flat_img_array):
    end_char = ",\n" if (i+1) % 10 == 0 else ", "  # 每10个数换行
    print(f"{value:.6f}", end=end_char)
print("};")

# 推理
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 获取输出（注意：输出可能有多个，我们取第一个 'ld_21_3d'）
output_data = interpreter.get_tensor(output_details[0]['index'])  # shape=(1,63)

# 解析21个关键点的3D坐标（x/y/z）
landmarks = output_data[0].reshape(-1, 3)  # 转换为 (21,3) 数组
print("Hand Landmarks (x, y, z):")
for i, (x, y, z) in enumerate(landmarks):
    print(f"Point {i}: ({x:.1f}, {y:.1f}, {z:.1f})")

# 可选：可视化2D关键点（忽略z坐标）
import matplotlib.pyplot as plt
plt.imshow(image)
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10)
plt.savefig("hand_with_landmarks.png")
#plt.show()
