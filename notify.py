#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 reports/latest.md 推送到手机.
配置了哪个渠道就发哪个 (环境变量 / GitHub Secrets):
  微信:  SERVERCHAN_KEY            (Server酱 https://sct.ftqq.com, 微信扫码即得, 免费额度每天 5 条)
  邮件:  SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASS MAIL_TO
"""
import os, re, smtplib, ssl
from email.mime.text import MIMEText
from email.header import Header
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE, "reports", "latest.md")
with open(path, encoding="utf-8") as f:
    body = f.read()

# 标题: 把两个币的结论提出来, 通知栏一眼看到
heads = re.findall(r"^## (\w+)\s+—\s+(.+?)\s+——", body, flags=re.M)
title = " | ".join(f"{c} {a}" for c, a in heads) or "每日操作建议"
need_action = any(a.startswith(("买入", "卖出")) for _, a in heads)
title = ("🔔 " if need_action else "✅ ") + title

sent = []

key = os.environ.get("SERVERCHAN_KEY", "").strip()
if key:
    r = requests.post(f"https://sctapi.ftqq.com/{key}.send",
                      data={"title": title[:32], "desp": body}, timeout=20)
    print("Server酱:", r.status_code, r.text[:200])
    sent.append("wechat")

host = os.environ.get("SMTP_HOST", "").strip()
if host:
    user, pw, to = os.environ["SMTP_USER"], os.environ["SMTP_PASS"], os.environ.get("MAIL_TO") or os.environ["SMTP_USER"]
    port = int(os.environ.get("SMTP_PORT", "465"))
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    msg["From"], msg["To"] = user, to
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as s:
            s.login(user, pw); s.sendmail(user, [to], msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls(context=ssl.create_default_context()); s.login(user, pw); s.sendmail(user, [to], msg.as_string())
    print("邮件已发送至", to)
    sent.append("email")

if not sent:
    print("未配置任何通知渠道 (SERVERCHAN_KEY 或 SMTP_*), 仅生成报告.")
    print(body)
