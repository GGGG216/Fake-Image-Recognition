import requests
import os
import pandas as pd
import time
import random
import json
from datetime import datetime, timedelta

class MultiKeyImageDownloader:
    def __init__(self):
        # 创建指定的目录结构
        self.base_dir = "data_SD_50test"
        self.image_dir = os.path.join(self.base_dir, "real_images")
        
        # 确保目录存在
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 10个Unsplash Access Keys - 请替换为您的实际密钥
        self.access_keys = [
            "YF2uAGv8I1xP-IL2PZwtjhJAwND0FCTWDtHnzFgeVsA",  # 您的第一个密钥
            "uIpbhXdzO3CtU61EA-xNpYGZJ_T-0uAev3vRDfzEKCs",
            "f6WA0ychcT7KArQ9AmJ_tD_FIkSWzgumN3f4C1D0eTY", 
            "Dw4vc_9zou_4wtY2tkW2sLqJb8UpzzfOrng3gG2IGl0",
            "HHLrREULnP8ngtVys1F_9etBud89H6WyYMMD0ZVJ5y8",
            "YZ05BcHwhAzlPAHEPq6a9gK4lEedC6h-jMDW0MBu6y4",
            "FXdZEFk9AmOd9_JmrRJ3ATjSwmBMZsiK3ancfu1kP4c",
            "oBhUUczx30L7rDnpW8F30z0gq00ZnDimjWZajPxGnqs",
            "ee5DX1kr6xVXiLHKf754Xr4v9Qai5ZCkMZk0-tgWJmQ",
            "7oqb2qFvIeQsPs12xjVKTITNVPl_aNWmPeS4oNnTPZw"
        ]
        
        # API配置
        self.api_configs = []
        for i, key in enumerate(self.access_keys):
            self.api_configs.append({
                'name': f'unsplash_{i+1}',
                'access_key': key,
                'search_url': "https://api.unsplash.com/search/photos",
                'rate_limit': 50,  # 每小时限制
                'requests_made': 0,
                'last_reset': datetime.now(),
                'enabled': True,
                'fail_count': 0
            })
        
        # 状态跟踪
        self.download_log = os.path.join(self.base_dir, "download_log.json")
        self.load_progress()
        
    def load_progress(self):
        """加载下载进度"""
        if os.path.exists(self.download_log):
            with open(self.download_log, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
            print(f"📁 加载已有进度: {self.progress['total_downloaded']} 张图片已下载")
        else:
            self.progress = {
                'downloaded_prompts': [],
                'failed_prompts': [],
                'total_downloaded': 0,
                'current_prompt_index': 0,
                'start_time': datetime.now().isoformat(),
                'api_usage': {config['name']: 0 for config in self.api_configs}
            }
            print("🆕 创建新的下载进度文件")
    
    def save_progress(self):
        """保存下载进度"""
        with open(self.download_log, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def can_make_request(self, api_config):
        """检查是否可以发出API请求"""
        now = datetime.now()
        
        # 检查是否需要重置计数器（每小时重置）
        if now - api_config['last_reset'] > timedelta(hours=1):
            api_config['requests_made'] = 0
            api_config['last_reset'] = now
            api_config['fail_count'] = 0
            print(f"🔁 {api_config['name']} API限制已重置")
        
        return api_config['requests_made'] < api_config['rate_limit'] and api_config['enabled']
    
    def get_available_api(self):
        """获取可用的API"""
        available_apis = [api for api in self.api_configs if self.can_make_request(api)]
        
        if not available_apis:
            # 所有API都达到限制，计算最近的重置时间
            reset_times = []
            for api in self.api_configs:
                next_reset = api['last_reset'] + timedelta(hours=1)
                wait_seconds = (next_reset - datetime.now()).total_seconds()
                if wait_seconds > 0:
                    reset_times.append(wait_seconds)
            
            if reset_times:
                wait_time = min(reset_times)
                print(f"⏳ 所有API都达到限制，等待 {wait_time/60:.1f} 分钟...")
                time.sleep(min(wait_time, 3600))  # 最多等待1小时
            else:
                print("⏳ 所有API都达到限制，等待1小时...")
                time.sleep(3600)
            
            return self.get_available_api()
        
        # 优先选择使用次数少的API
        available_apis.sort(key=lambda x: self.progress['api_usage'].get(x['name'], 0))
        return available_apis[0]
    
    def search_unsplash(self, api_config, prompt):
        """使用Unsplash API搜索图片"""
        headers = {
            "Authorization": f"Client-ID {api_config['access_key']}",
            "Accept-Version": "v1"
        }
        
        params = {
            "query": prompt,
            "per_page": 1,  # 每次只请求1张以减少API调用
            "orientation": "landscape"
        }
        
        try:
            response = requests.get(api_config['search_url'], headers=headers, params=params, timeout=30)
            api_config['requests_made'] += 1
            
            # 更新API使用统计
            self.progress['api_usage'][api_config['name']] = self.progress['api_usage'].get(api_config['name'], 0) + 1
            
            if response.status_code == 200:
                data = response.json()
                return data['results'][0] if data['results'] else None
            elif response.status_code == 403:
                print(f"❌ {api_config['name']} API限制已达到，暂时禁用")
                api_config['enabled'] = False
                return None
            else:
                print(f"❌ {api_config['name']} API错误: {response.status_code}")
                api_config['fail_count'] += 1
                if api_config['fail_count'] >= 5:
                    api_config['enabled'] = False
                    print(f"🚫 {api_config['name']} 因多次失败被禁用")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ {api_config['name']} 请求超时")
            api_config['fail_count'] += 1
            return None
        except Exception as e:
            print(f"❌ {api_config['name']} 请求异常: {e}")
            api_config['fail_count'] += 1
            return None
    
    def download_image(self, image_url, filename):
        """下载图片文件"""
        try:
            img_response = requests.get(image_url, stream=True, timeout=60)
            if img_response.status_code == 200:
                filepath = os.path.join(self.image_dir, filename)
                with open(filepath, 'wb') as f:
                    for chunk in img_response.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"  下载错误: {e}")
        return False
    
    def clean_filename(self, text):
        """清理文件名，移除非法字符"""
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            text = text.replace(char, '_')
        return text[:100]  # 限制文件名长度
    
    def download_images_continuously(self):
        """持续下载图片"""
        try:
            # 读取Excel文件中的B列
            df = pd.read_excel("Prompt.xlsx")
            prompts = df.iloc[:, 1].tolist()  # B列是第1列（0-indexed）
            print(f"✅ 成功读取 {len(prompts)} 个提示词")
            print(f"📝 前5个提示词: {prompts[:5]}")
        except Exception as e:
            print(f"❌ 读取Excel文件失败: {e}")
            print("💡 请确保 Prompt.xlsx 文件存在且B列包含提示词")
            return
        
        total_prompts = len(prompts)
        start_index = self.progress['current_prompt_index']
        
        print(f"🎯 开始下载，从第 {start_index + 1} 个提示词开始")
        print(f"🔑 可用API密钥: {len([api for api in self.api_configs if api['enabled']])}/{len(self.api_configs)}")
        
        for i in range(start_index, total_prompts):
            prompt = prompts[i]
            prompt_str = str(prompt).strip()
            
            if not prompt_str or prompt_str in self.progress['downloaded_prompts']:
                print(f"⏭️ 跳过空提示词或已下载: '{prompt_str}'")
                self.progress['current_prompt_index'] = i + 1
                self.save_progress()
                continue
            
            print(f"\n🔍 搜索 ({i+1}/{total_prompts}): '{prompt_str}'")
            
            # 获取可用API
            api_config = self.get_available_api()
            print(f"  使用 {api_config['name']} (已用: {api_config['requests_made']}/{api_config['rate_limit']})")
            
            # 搜索图片
            photo = self.search_unsplash(api_config, prompt_str)
            
            if photo:
                image_url = photo['urls']['regular']
                # 清理文件名
                clean_prompt = self.clean_filename(prompt_str)
                filename = f"{i+1:04d}_{clean_prompt}.jpg"
                
                # 下载图片
                if self.download_image(image_url, filename):
                    print(f"  ✅ 下载成功: {filename}")
                    self.progress['downloaded_prompts'].append(prompt_str)
                    self.progress['total_downloaded'] += 1
                else:
                    print(f"  ❌ 下载失败")
                    self.progress['failed_prompts'].append(prompt_str)
            else:
                print(f"  ⚠️ 未找到图片")
                self.progress['failed_prompts'].append(prompt_str)
            
            # 更新进度
            self.progress['current_prompt_index'] = i + 1
            self.save_progress()
            
            # 显示统计信息
            enabled_apis = len([api for api in self.api_configs if api['enabled']])
            print(f"  📊 进度: {self.progress['total_downloaded']}/{total_prompts} | 可用API: {enabled_apis}")
            
            # 随机延迟避免被检测为机器人
            delay = random.uniform(3, 8)
            time.sleep(delay)
        
        print(f"\n🎉 下载完成！")
        print(f"✅ 成功下载: {self.progress['total_downloaded']}/{total_prompts}")
        print(f"❌ 失败: {len(self.progress['failed_prompts'])}")
        print(f"📁 图片保存在: {self.image_dir}")
        
        # 显示API使用统计
        print("\n📈 API使用统计:")
        for api_name, usage in self.progress['api_usage'].items():
            print(f"  {api_name}: {usage} 次")

def main():
    downloader = MultiKeyImageDownloader()
    
    print("🚀 Unsplash多密钥图片下载器")
    print("=" * 50)
    print(f"📁 工作目录: {os.getcwd()}")
    print(f"📁 图片保存到: {downloader.image_dir}")
    print(f"🔑 配置密钥数: {len(downloader.access_keys)}")
    print("=" * 50)
    
    # 验证密钥
    valid_keys = 0
    for key in downloader.access_keys:
        if key and not key.startswith("YOUR_") and key != "YF2uAGv8I1xP-IL2PZwtjhJAwND0FCTWDtHnzFgeVsA":
            valid_keys += 1
    
    print(f"🔑 有效密钥: {valid_keys}/{len(downloader.access_keys)}")
    
    if valid_keys == 0:
        print("❌ 请先在代码中配置您的Unsplash Access Keys")
        return
    
    try:
        downloader.download_images_continuously()
    except KeyboardInterrupt:
        print("\n⏸️ 下载被用户中断，进度已保存")
        print(f"💡 重新运行程序将从上次中断处继续")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("进度已保存，可以重新运行继续下载")

if __name__ == "__main__":
    main()