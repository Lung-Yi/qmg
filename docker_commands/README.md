# 使用Docker虛擬環境執行multi-gpu加速量子電路模擬
1. 可先參考nvidia整合qiskit並開發的`cusvaer`套件，目前(Aug. 2024)只支援使用docker虛擬環境使用:
https://docs.nvidia.com/cuda/cuquantum/latest/appliance/qiskit.html

2. 接著至以下網站選擇適當的docker images做下載:
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance/tags

## 下載docker images 並且執行容器
1. `docker pull (REPOSITORY):(TAG)`
2. `docker image ls` # 檢查是否下載成功
3. `docker run -itd --gpus all (REPOSITORY):(TAG)` # 如果沒加--gpus all 的話，就不會使用到gpu
4. `docker ps` # 檢查是否有容器產生
5. `docker exec -it (CONTAINER ID) /bin/bash` # 進入容器內，即可開始操作

## 複製文件
1. 
```
docker cp mycontainer:/opt/file.txt /opt/local/ # 從container複製出去外面
docker cp /opt/local/file.txt mycontainer:/opt/ # 從外面複製進去container
```
2. Docker Image存檔出一個檔案
`docker save -o (filename).tar (image name:tag)`
3. 把檔案 Load 到 Docker 的指令如下
`docker load -i (filename).tar`

## 停止並刪除容器
1. `docker ps -a` # 檢查所有容器
2. `docker stop (CONTAINER ID)` # 停止指定容器
3. `docker rm (CONTAINER ID)` # 刪除指定容器
4. `docker rmi (IMAGE ID)` # 刪除指定image

