#### 文件目录
```
MLP
│─── run.py 程序入口
│─── config 配置文件夹
│    │─── genotype 模型结构文件夹，训练时的模型结构从这里读取
│    │    │─── cifar-config.json
│    │    └─── UCI-config.json
│    │─── search-config 搜索配置
│    │    │─── cifar-config.json
│    │    └─── UCI-config.json
│    │─── split-file 划分测试集文件
│    │    │─── cifar10-split.txt
│    │    │─── cifar100-split.txt
│    │    │─── HAPT-split.txt
│    │    └─── UJI-split.txt
│    │─── train-config 训练配置
│    │    │─── cifar-config.json
│    │    └─── UCI-config.json
│─── data 数据
│─── log 日志
│─── model 模型
│    │─── operation 被搜索的操作
│    │    │─── classify_opt.py MLP操作
│    │    └─── cnn_opt.py CNN操作
│    │─── classifier_model.py 搜索时、训练时使用的分类模型
│    │─── pre_model.py 前置模型，CNN（NasNet）MLP（维度对齐）
│    │─── random_foreset.py 随机森林
│    │─── train.py 搜索、训练执行
│    └─── train_model.py 拼接pre_model 和 classifier_model构造的模型
│─── report 报告
│    │─── MLP.pptx 结构图
│    └─── 算法说明.pdf
└─── untils 工具
     │─── data_process.py 数据处理
     │─── flop_becnmark.py 参数估计
     └─── util.py 日志、准确率、模型存取工具
```
程序入口：
run.py
执行不同操作只需要修改 run.py 18行，更改使用的配置文件夹即可。