@echo off
REM Windows: 用"任务计划程序"每天 21:00 运行本文件
cd /d %~dp0
python -W ignore advisor.py advise
pause
