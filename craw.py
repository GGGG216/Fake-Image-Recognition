import requests
import os
import pandas as pd
import time
from urllib.parse import quote

def download_images_safely():
    """
    安全使用Unsplash API下载图片
    只需要Access Key (Client ID)
    """
    
    # 创建保存图片的文件夹
    image_dir = "real_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # 读取Excel文件
    try:
        df = pd.read_excel("A230.1_BatchDataSheets.xlsx",header=None)
        prompts = df.iloc[:, 1].tolist()
        print(f"成功读取 {len(prompts)} 个提示词")
        print("提示词列表:", prompts)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    
    # 替换成您实际的Unsplash Access Key
    UNSPLASH_ACCESS_KEY = "YF2uAGv8I1xP-IL2PZwtjhJAwND0FCTWDtHnzFgeVsA"
    
    if UNSPLASH_ACCESS_KEY == "YOUR_ACTUAL_ACCESS_KEY_HERE":
        print("请先获取Unsplash Access Key：")
        print("1. 访问 https://unsplash.com/oauth/applications")
        print("2. 创建New Application (选择Demo类型)")
        print("3. 复制Access Key到代码中")
        return
    
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1"
    }
    
    downloaded_count = 0
    
    for i, prompt in enumerate(prompts):  # 先测试前5个
        prompt_str = str(prompt).strip()
        print(f"搜索: '{prompt}' ({i+1}/{min(5, len(prompts))})")
        
        try:
            search_url = "https://api.unsplash.com/search/photos"
            params = {
                "query": prompt_str,
                "per_page": 3,  # 每次2张
                "orientation": "landscape"
            }
            
            response = requests.get(search_url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['results']:
                    for j, photo in enumerate(data['results'][:1]):  # 只下载1张测试
                        img_url = photo['urls']['regular']
                        filename = f"{i+1:02d}_{prompt[:10]}.jpg"
                        filepath = os.path.join(image_dir, filename)
                        
                        img_response = requests.get(img_url, stream=True)
                        if img_response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                for chunk in img_response.iter_content(1024):
                                    f.write(chunk)
                            
                            print(f"  ✅ 下载: {filename}")
                            downloaded_count += 1
                        else:
                            print(f"  ❌ 下载失败")
                
                else:
                    print(f"  ⚠️ 未找到图片")
            else:
                print(f"  ❌ API错误: {response.status_code}")
                print(f"  💡 响应: {response.text[:100]}")
            
            #time.sleep(1)  # 礼貌延迟
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    print(f"\n测试完成！下载 {downloaded_count} 张图片")
    print("如果测试成功，可以修改代码处理全部提示词")

if __name__ == "__main__":
    download_images_safely()