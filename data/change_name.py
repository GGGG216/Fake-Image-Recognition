import os
import shutil
import re

def rename_images():
    # 定义文件夹路径
    source_folder = 'data_SD_50test/real_images'
    target_folder = 'data_SD_50test/real_images_rename'
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在！")
        return
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.jpg'):
            # 使用正则表达式匹配数字部分
            match = re.match(r'^(\d+)_', filename)
            
            if match:
                original_number = match.group(1)
                
                try:
                    # 将数字转换为整数并+1
                    new_number = int(original_number) + 1
                    
                    # 格式化为4位数字，前面补零
                    new_number_str = str(new_number)
                    
                    # 构建新的文件名
                    new_filename = f"{new_number_str}.jpg"
                    
                    # 构建完整的文件路径
                    source_path = os.path.join(source_folder, filename)
                    target_path = os.path.join(target_folder, new_filename)
                    
                    # 复制并重命名文件
                    shutil.copy2(source_path, target_path)
                    print(f"已处理: {filename} -> {new_filename}")
                    
                except ValueError:
                    print(f"警告：无法解析文件 '{filename}' 中的数字，跳过此文件")
            else:
                print(f"警告：文件 '{filename}' 不符合命名格式，跳过此文件")
    
    print(f"\n处理完成！所有文件已保存到 '{target_folder}' 文件夹中")

if __name__ == "__main__":
    rename_images()