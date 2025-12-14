import pandas as pd
import random
from openpyxl import Workbook

# 定义关键词库
scenes = [
    "山脉", "海洋", "森林", "沙漠", "草原", "湖泊", "河流", "瀑布", 
    "冰川", "山顶", "丘壑", "海湾", "山丘", "山地", "海滩", "岛屿", 
    "峡谷", "洞穴", "田野", "湿地", "雪山", "丛林", "悬崖", "平原"
]

# 白天天气
day_weathers = [
    "晴天", "阴天", "多云", "微风", "薄雾", "细雨", "暴风雪", "大风", "暴雨"
]

# 夜晚天气
night_weathers = [
    "晴朗", "多云", "微风", "薄雾", "夜空", "月亮"
]

# 白天时间
day_times = [
    "清晨", "早晨", "上午", "中午", "下午", "黄昏"
]

# 夜晚时间  
night_times = [
    "傍晚", "夜晚", "午夜"
]

seasons = ["春天", "夏天", "秋天", "冬天"]

def get_compatible_weather(time):
    """根据时间返回兼容的天气"""
    if time in day_times:
        # 白天：可以使用白天天气和特殊天气
        return random.choice(day_weathers)
    elif time in night_times:
        # 夜晚：可以使用夜晚天气和特殊天气
        return random.choice(night_weathers)
    else:
        return random.choice(day_weathers + night_weathers)

def get_compatible_time(weather):
    """根据天气返回兼容的时间"""
    if weather in day_weathers:
        return random.choice(day_times)
    elif weather in night_weathers:
        return random.choice(night_times)
    else:  # 特殊天气
        return random.choice(day_times + night_times)

def generate_prompt():
    """生成逻辑合理的单个提示词"""
    scene = random.choice(scenes)
    season = random.choice(seasons)
    
    # 随机选择生成策略：先定时间或先定天气
    if random.random() < 0.5:
        # 策略1：先随机时间，再选择兼容的天气
        time = random.choice(day_times + night_times)
        weather = get_compatible_weather(time)
    else:
        # 策略2：先随机天气，再选择兼容的时间
        weather = random.choice(day_weathers + night_weathers)
        time = get_compatible_time(weather)

    # 随机选择组合模板
    templates = [
        f"{scene}，{weather},{time}",
        f"{scene}，{weather}，{season}"
    ]
    
    return random.choice(templates)

def generate_1000_prompts():
    """生成1000个不重复的提示词"""
    prompts_set = set()
    
    print("正在生成1000个提示词...")
    while len(prompts_set) < 1000:
        prompt = generate_prompt()
        prompts_set.add(prompt)
        
        # 显示进度
        if len(prompts_set) % 100 == 0:
            print(f"已生成 {len(prompts_set)} 个提示词...")
    
    prompts_list = list(prompts_set)[:1000]
    print(f"成功生成 {len(prompts_list)} 个唯一提示词！")
    return prompts_list

def save_to_excel(prompts_list, filename="A230.1_BatchDataSheets.xlsx"):
    """保存到Excel文件，格式为A列空，B列提示词，C列横版"""
    # 创建DataFrame
    data = []
    for prompt in prompts_list:
        data.append({
            'A列': '',  # A列为空
            '提示词': prompt,  # B列为提示词
            'C列': '横版'  # C列为横版
        })
    
    df = pd.DataFrame(data)
    
    # 保存到Excel，不包含索引和表头
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        
        # 获取工作表并调整列宽
        worksheet = writer.sheets['Sheet1']
        worksheet.column_dimensions['A'].width = 10  # A列宽度
        worksheet.column_dimensions['B'].width = 50  # B列宽度（提示词列）
        worksheet.column_dimensions['C'].width = 10  # C列宽度
    
    print(f"提示词已保存到 {filename}")
    print("文件格式：")
    print("- A列：空")
    print("- B列：提示词")
    print("- C列：横版")

def main():
    """主函数"""
    try:
        # 生成1000个提示词
        prompts = generate_1000_prompts()
        
        # 保存到Excel
        save_to_excel(prompts, "A230.1_BatchDataSheets.xlsx")
        
        # 显示前10个生成的提示词作为示例
        print("\n前10个提示词示例：")
        for i, prompt in enumerate(prompts[:10], 1):
            print(f"{i:2d}. {prompt}")
            
        print(f"\n🎉 完成！已生成1000个提示词并保存到 A230.1_BatchDataSheets.xlsx")
        
    except Exception as e:
        print(f"生成过程中出现错误: {e}")

if __name__ == "__main__":
    main()