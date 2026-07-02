"""消息推送模块 — 支持 Bark / 企业微信机器人 / Server酱

环境变量（至少配一个）：
    export BARK_URL="https://api.day.app/yourkey"
    export WECOM_WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=..."
    export SERVERCHAN_KEY="your_sckey"

用法：
    from core.notify import send
    send("标题", "消息正文")
"""

import os
import json
import urllib.request
import urllib.error


def _send_bark(title: str, body: str) -> bool:
    """Bark iOS 推送 (https://github.com/Finb/Bark)"""
    url = os.environ.get("BARK_URL", "")
    if not url:
        return False
    try:
        data = json.dumps({"title": title, "body": body,
                           "group": "Turtle"}).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[notify] Bark 发送失败: {e}")
        return False


def _send_wecom(title: str, body: str) -> bool:
    """企业微信机器人 Webhook"""
    url = os.environ.get("WECOM_WEBHOOK", "")
    if not url:
        return False
    try:
        content = f"## {title}\n{body}"
        data = json.dumps({
            "msgtype": "markdown",
            "markdown": {"content": content}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[notify] 企业微信发送失败: {e}")
        return False


def _send_serverchan(title: str, body: str) -> bool:
    """Server酱 (https://sct.ftqq.com/)"""
    key = os.environ.get("SERVERCHAN_KEY", "")
    if not key:
        return False
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = json.dumps({"title": title, "desp": body}).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"[notify] Server酱发送失败: {e}")
        return False


# 渠道优先级
_CHANNELS = [
    ("Bark",       _send_bark),
    ("Server酱",   _send_serverchan),
    ("企业微信",    _send_wecom),
]


def send(title: str, body: str) -> bool:
    """按优先级尝试已配置的渠道发送通知，返回是否至少一个渠道成功"""
    any_ok = False
    for name, fn in _CHANNELS:
        if fn(title, body):
            print(f"[notify] ✅ {name} 发送成功")
            any_ok = True
            break
    if not any_ok:
        print("[notify] ❌ 无可用通知渠道，请配置 BARK_URL / WECOM_WEBHOOK / SERVERCHAN_KEY")
    return any_ok
