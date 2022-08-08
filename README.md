# 盘煤仪项目搭建

## 配置pip镜像源
```markdown
# windows环境(在C:\Users\XXX文件下创建pip文件夹，创建pip.ini文件)
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn

# Linux环境
## 1.切换目录，并新建 .pip 文件夹
    cd ~/.pip
    mkdir ~/.pip
## 2.进入 .pip 文件夹，并新建 pip.conf 文件
    cd ~/.pip
    touch pip.conf
## 3.编辑pip.conf
    vi pip.conf
    添加一下内容:
    [global] 
    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
    [install]
    trusted-host = https://pypi.tuna.tsinghua.edu.cn

## 查看配置
pip config list
```
## 安装依赖
```markdown
pip install -r requirement.txt
```
## 导出依赖包
pip freeze > requirements.txt
## 项目目录
```text
coal            #根目录
├─api           #API
├─core          #核心
├─models        #
├─schemas       #
├─static        #静态资源
├─venv          #虚拟环境
├─views         #视图
├─config.py     #配置类
├─main.py       #主函数
└─requirement.txt   #环境依赖
```


## 项目启动
```markdown
    uvicorn main:app --reload --host 0.0.0.0 --port 9000
```

```sh
    source /opt/python/venv/bin/activate
    进入虚拟环境目录
    nohup uvicorn main:app --host 0.0.0.0 --port 8001 --reload  > coal.log 2>&1 &
    后台启动
```
```sh
lsof -i:8000
kill -9 PID
```
## 其他

