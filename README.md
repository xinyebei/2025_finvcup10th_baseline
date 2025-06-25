<!--
 * @description: 
 * @param : 
 * @return: 
 * @Author: xinyebei@xinye.com
 * @Date: 2025-04-28 14:35:42
 * @LastEditors: maoyangjun@xinye.com
-->
> # 第十届信也杯baseline
这是第十届信也科技杯baseline。 本届全球算法大赛聚焦人脸深度鉴伪，挑战前沿AI技术，以推动国际合作，引导科技向善。



## 基础模型baseline
<!-- 表格 -->
表中指标为auc
|           测试集\模型       |    xception    | ucf | srm |  clip-base| xception+sbi|
| :--------------------------: | :--------: | :----:  | :----: |:----: | :----: | 
| 公开测试集 | 0.660 | 0.668 | 0.681 |0.628 |   0.732  |


上述模型缩写请见[DeepFakeBenchmark](https://github.com/SCLBD/DeepfakeBench)


## 1. 数据准备
将训练、公开测试集解压到`/data`下,包含`train.txt`文件，整理构成
``` bash
├── data
│   ├── data1    #训练集+公开测试集，在压缩包内
│   │    ├── train
│   │    │   └── images
│   │    ├── test1
│   │    │   └── images
│   │    └── train.txt
│   │  
│   ├── data2   #私有测试集目录，未给出，在复赛时私有测试集会被挂载到该路径，选手可以忽略
│   │    └── test2
│   │        └── images
│   │ 
│  
│        
├── algorithm

```

- train.txt 文件中包含了所有的训练样本，以`file_name label` 形式存储，其中label为0表示真实样本，为1表示伪造样本


## 2. Environments
### 2.1. 在本地运行镜像 

我们推荐各位参赛者在docker中运行、调试代码，最终的提交代码会要求在docker中运行。选手可以从[这个链接](https://docker.aityp.com/r/docker.io/pytorch/pytorch) 或大赛官方镜像仓库(`详细见大赛操作手册`)拉取`pytorch`镜像，将拉取的镜像作为基础镜像，把本地代码挂载为`/algorithm`，以下是一个简单的例子。

**注意：**     `xxxxx-runtime`镜像内部`不`包含nvcc环境， `xxxxx-devel`镜像内部包含nvcc环境。如果参赛者使用的模型，在pip安装包时需要编译新算子，需要使用`xxxxx-devel`镜像，否则使用`xxxxx-runtime`镜像即可。**请不要选择大于12.4 cuda版本的镜像，如算法必须使用高于12.4版本的cuda，请联系大赛官方 [xinyebei@xinye.com](xinyebei@xinye.com**)**。
``` bash
#从大赛官方仓库拉取镜像
docker login --username=team35@1772449545059402  finvcup-registry.cn-shanghai.cr.aliyuncs.com 
#密码: FQ1K3@XJev}(p!fzC6Ws7erBYXVAdzzu

docker pull finvcup-registry.cn-shanghai.cr.aliyuncs.com/finvcup/official:torch2.5.1-cuda12.4-cudnn9-runtime



#从第三方源拉取镜像
#拉取镜像，参赛者可根据自己的需求拉取不同的cuda\pytorch版本的镜像
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

docker tag  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime  docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime



# 启动容器并置于后台，挂载本地代码到docker的/algorithm目录下
docker run -d -it --ipc host --gpus all \ 
-v /path/to/your/code:/algorithm \       
-v /path/to/your/data:/data \
--name cuda_runtime  \
-i docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime  \
tail -f /dev/null
# 启动完成后可以通过docker ps -a查看已启动的容器


# 手动进入容器
docker exec -it cuda_runtime /bin/bash
apt-get update && apt-get install build-essential
pip install -r /algorithm/requirements.txt
#vscode用户，可以直接在docker内开发调试，参考：https://zhuanlan.zhihu.com/p/496213879 
```

### 2.2. 大赛镜像提交(仅针对晋级复赛的参赛队伍)
大赛规定，选手需要提交`Dockerfile`作为构建镜像的依据，选手需要保证镜像中包含所有运行代码所需的环境，包括但不限于：Python环境、依赖库、模型文件等。选手需要将`Dockerfile`和构建镜像所需的文件打包成一个压缩包，提交到大赛平台。请参考`./Dockerfile`编写, 使用`docker build -t xxx:latest .` 构建镜像，注意通过如下代码安装额外依赖:
```bash
COPY ./requirements.txt ${HOME}/requirements.txt

# pip 更换为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装依赖
RUN pip install -r requirements.txt
```

**大赛提供的官方镜像即由上述Dockfile构建完成**

## 3. 预处理数据
我们借鉴了[DeepFakeBenchmark](https://github.com/SCLBD/DeepfakeBench)的预处理数据方式，见本仓库AlignFacePreprocess类，原始代码见`https://github.com/SCLBD/DeepfakeBench?tab=readme-ov-file#3-preprocessing-optional`，感谢原作者的贡献。
``` bash

python preprocess_img.py --image_path /data/data1/train/images  --save_path /data/data1/train/images_crop

#如有必要，请选手自行划分验证集并预处理
# python preprocess_img.py --image_path /data/data1/val/images  --save_path /data/data1/val/images_crop


#如果觉得速度太慢可以使用如下多进程脚本
python preprocess_img_multiprocess.py --image_path /data/data1/train/images  --save_path /data/data1/train/images_crop --total_splits 4

#验证集...
#测试集...

```


## 4. 训练
```bash
# 生成训练metainfo文件
python prepare_dataset_info.py --label_info /data/data1/train.txt --mode train --dataset_name train --dataset_root_path /data/data1/train/images_crop/crop

# 如有必要请选手自行划分验证集，并模仿train.txt 格式生成val.txt
# python prepare_dataset_info.py --label_info /data/data1/val.txt --mode val --dataset_name val --dataset_root_path /data/data1/val/images_crop/crop



#将生成的dataset_name   “train”   添加到configs/xception.yaml中  train_dataset:[train,], val添加至文件中 test_dataset:[val,]中
#模型权重文件  xception-b5690688.pth 可以从 https://data.lip6.fr/cadene/pretrainedmodels/ 下载

# 单卡训练
python train.py --task_target exp_name --detector_path ./configs/xception.yaml 

# ddp训练
bash train.sh exp_name 48001 ./configs/xception.yaml 
```

## 5. 推理(仅针对晋级复赛的参赛队伍)
```bash
#example_input.csv中有一些示例图片，请将test1测试集中的所有图片先写入某个 .csv文件中, 注意下列推理脚本中， input_csv中输入的图片是未经过预处理的原图，在predictor.py中会进行预处理
python predictor.py --input_csv example/example_input.csv --output_csv example/example_output.csv
bash run.sh  # run.sh 中仅有一行命令，即上述predictor.py的命令



#或者 选手可以自行编排流程(预处理图片，推理，后处理结果 等等)写入run.sh中，请注意私有测试集的路径是/data/data2/test2/images   
# 预处理.....   python preprocess_multiprocess.py --image_path /data/data2/test2/images  --save_path /data/data2/test2/images_crop --total_splits 4
# 推理.....     python predict.py
# 后处理.....   python postprocess.py
bash run.sh 

```

## 6. 注意事项
1. 训练时，请确保`configs/xxx.yaml`中的`train_dataset`字段与`data/prepare_dataset_info.py`生成的`dataset_name`一致，这样才可以训练。
2. 请各位选后自行组织自己的推理脚本，并写入`run.sh`中，但是注意私有测试集的路径是`/data/data2/test2/images`，选手需要确保脚本可以顺利对改路径下的图片进行推理。
3. `submit.csv`文件需要包含"image_name, score" header，csv文件使用`,`作为分隔符。



## Acknowledgement
本项目依据于[DeepFakeBenchmark](https://github.com/SCLBD/DeepfakeBench)构建，感谢原作者杰出的贡献，如有需要，请移步原仓库star，citations.