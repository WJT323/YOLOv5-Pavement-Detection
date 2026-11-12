import torch

# 1. 加载你训练好的模型
model_path = 'weights/best.pt'
ckpt = torch.load(model_path, map_location='cpu')

# 2. 打印看看模型里现在存的是什么牛鬼蛇神
print("修复前的类别名：", ckpt['model'].names)

# 3. 根据你之前的 yaml 文件，把正确的中文写进去
correct_names = {
    0: '坑槽', 1: '松散', 2: '横向裂纹', 3: '纵向裂纹', 
    4: '网状裂纹', 5: '横向修补', 6: '纵向修补', 7: '块状修补',
    8: '网状修补', 9: '标线模糊', 10: '抛洒物', 11: '锥形桶',
    12: '圆形桶', 13: '人', 14: '方形雨水盖', 15: '圆形雨水盖',
    16: '圆形井盖', 17: '护栏损坏'
}

# 4. 强制覆盖模型的类别名
ckpt['model'].names = correct_names

# 5. 保存为一个新的模型文件
new_model_path = 'weights/best_fixed.pt'
torch.save(ckpt, new_model_path)
print(f"修复完成！新模型已保存至: {new_model_path}")