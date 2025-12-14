import os
import shutil
import re

def merge_and_rename_images():
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 目标文件夹路径
    source_folder = os.path.join(current_dir, "data_SD_50test")
    target_folder = os.path.join(current_dir, "merged_images")
    
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 正则表达式匹配文件夹名称格式：generated_image_开始数字_结束数字
    pattern = re.compile(r'generated_image_(\d+)_(\d+)')
    
    # 遍历源文件夹中的所有项目
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        
        # 只处理文件夹且名称符合pattern
        if os.path.isdir(item_path) and pattern.match(item):
            match = pattern.match(item)
            start_num = int(match.group(1))  # 起始编号
            # end_num = int(match.group(2))   # 结束编号（这里用不到）
            
            print(f"处理文件夹: {item}")
            
            # 遍历该文件夹中的所有文件
            for filename in os.listdir(item_path):
                file_path = os.path.join(item_path, filename)
                
                # 只处理图片文件（可以根据需要扩展支持的格式）
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # 提取文件名的数字部分
                    file_num_match = re.search(r'(\d+)', filename)
                    if file_num_match:
                        file_num = int(file_num_match.group(1))
                        
                        # 计算实际的文件编号：起始编号 + 文件在当前文件夹中的序号 - 1
                        actual_num = start_num + file_num - 1
                        
                        # 获取文件扩展名
                        _, ext = os.path.splitext(filename)
                        
                        # 构建新的文件名
                        new_filename = f"{actual_num}{ext}"
                        new_file_path = os.path.join(target_folder, new_filename)
                        
                        # 复制文件到目标文件夹并重命名
                        shutil.copy2(file_path, new_file_path)
                        print(f"  已复制: {filename} -> {new_filename}")
    
    print(f"\n所有图片已合并到: {target_folder}")

# 更健壮的版本，处理可能的文件名冲突和其他情况
def merge_and_rename_images_robust():
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 目标文件夹路径
    source_folder = os.path.join(current_dir, "data_SD_50test")
    target_folder = os.path.join(current_dir, "merged_images")
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹 {source_folder} 不存在!")
        return
    
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 正则表达式匹配文件夹名称格式
    pattern = re.compile(r'generated_image_(\d+)_(\d+)')
    
    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
    
    # 统计信息
    total_files = 0
    processed_folders = 0
    
    # 先收集所有符合条件的文件夹并排序
    folders_to_process = []
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)
        if os.path.isdir(item_path) and pattern.match(item):
            match = pattern.match(item)
            start_num = int(match.group(1))
            folders_to_process.append((item, start_num))
    
    # 按起始编号排序
    folders_to_process.sort(key=lambda x: x[1])
    
    # 处理每个文件夹
    for folder_name, start_num in folders_to_process:
        folder_path = os.path.join(source_folder, folder_name)
        print(f"处理文件夹: {folder_name} (起始编号: {start_num})")
        
        processed_folders += 1
        
        # 获取文件夹中的所有图片文件并排序
        image_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    image_files.append(filename)
        
        # 按文件名排序（确保顺序正确）
        image_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        
        # 处理每个图片文件
        for i, filename in enumerate(image_files, 1):
            file_path = os.path.join(folder_path, filename)
            
            # 计算实际的文件编号
            actual_num = start_num + i - 1
            
            # 获取文件扩展名
            _, ext = os.path.splitext(filename)
            
            # 构建新的文件名
            new_filename = f"{actual_num}{ext}"
            new_file_path = os.path.join(target_folder, new_filename)
            
            # 如果目标文件已存在，先删除（避免覆盖问题）
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            
            # 复制文件到目标文件夹并重命名
            shutil.copy2(file_path, new_file_path)
            print(f"  已复制: {filename} -> {new_filename}")
            total_files += 1
    
    print(f"\n处理完成!")
    print(f"共处理 {processed_folders} 个文件夹")
    print(f"共合并 {total_files} 个图片文件")
    print(f"目标文件夹: {target_folder}")

if __name__ == "__main__":
    # 使用更健壮的版本
    merge_and_rename_images_robust()