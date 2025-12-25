import requests
import json
try:
    from config.account_config import WechatConfig
except ImportError:
    pass

class WechatBot:
    def __init__(self, key=None):
        # 优先用传入的key，否则读配置
        self.key = key if key else getattr(WechatConfig, 'WEBHOOK_KEY', "")
        self.url = f'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.key}'

    def send_text(self, content):
        if not self.key:
            print("⚠️ 微信 Key 未配置，跳过发送")
            return
            
        headers = {'Content-Type': 'application/json'}
        data = {
            "msgtype": "text",
            "text": { "content": content }
        }
        try:
            resp = requests.post(self.url, headers=headers, json=data)
            if resp.json().get('errcode') == 0:
                print(f"✅ 微信发送成功: {content[:10]}...")
            else:
                print(f"❌ 微信发送失败: {resp.text}")
        except Exception as e:
            print(f"❌ 微信发送异常: {e}")