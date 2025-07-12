# 1 安装环境
apt-get update 
apt-get install git-lfs curl aria2
# 2 安装脚本并配置环境变量
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
echo "export PATH=\"\$PATH:$(pwd)\" && alias hfd=\"$(pwd)/hfd.sh\"" >> ~/.bashrc && source ~/.bashrc
