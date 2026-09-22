#!/usr/bin/env bash
# 每晚北京时间 21:00 运行. crontab -e 加一行:
#   0 21 * * * /bin/bash /path/to/crypto_advisor/run.sh >> /path/to/crypto_advisor/reports/cron.log 2>&1
cd "$(dirname "$0")"
python3 -W ignore advisor.py advise
