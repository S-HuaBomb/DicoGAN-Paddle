# DiscoGAN-Paddle

## 简介

Re-implement DiscoGAN in Paddle

基于 pytorch 源码：https://github.com/SKTBrain/DiscoGAN

![](./assets/discoGAN.jpg)

DiscoGAN 通过 GAN 来学习不同域的特征，捕捉不同域之间的关系，利用这些关系，可以生成将风格从一个域转移到另一个域的图片，同时保留了关键属性，如方向和面部识别特征。

论文链接：[Learning to Discover Cross-Domain Relations with Generative Adversarial Networks.](https://arxiv.org/pdf/1703.05192.pdf)

## 环境依赖

- python 3.7
- paddle 2.1

## 快速开始

预训练模型、训练日志、测试集demo，可到网盘自取：链接：https://pan.baidu.com/s/1hB34lEKMdh4YIh84kQOILQ 
提取码：sbsi

### Train

对于 gender conversion，训练 50000 个 iters 即可达到最佳效果。

#### 数据预处理（可选）

将 celeba 数据集 croped 到 128×128：

```
python DiscoGAN-Paddle/crop_celeba.py celeba_cropped path/to/img_align_celeba/ -o path/to/croped_celeba
```

下面的运行训练和预测代码以性别转换生成为例，

- **单卡：**
    
    ```
    python DiscoGAN-Paddle/discogan/image_translation.py \
        --image_dir path/to/img_align_celeba \
        --attr_file path/to/list_attr_celeba.txt \
        --task_name celebA \
        --style_A Male \
        --n_test 200 \
        --batch_size 200 \
        --epoch_size 1000 \
        --result_path ./results \
        --model_path ./models \
        --log_out ./logs \
        --ckpt_path path/to/discoGAN.pdparams \
        --local_rank -1 \
        --num_workers 4 \
        --learning_rate 0.0002
    ```
    
- **单机 4 卡：**

    AI Studio 脚本训练：`train.sh`
    
    ```shell script
    #!/bin/bash
    
    CELEBA=/root/paddlejob/workspace/train_data/datasets/data107578/img_align_celeba.zip
    ATTR_TXT=/root/paddlejob/workspace/train_data/datasets/data107578/list_attr_celeba.txt
    
    TO_DIR=/root/paddlejob/workspace/train_data/datasets/
    IMG_CELEBA=/root/paddlejob/workspace/train_data/datasets/img_align_celeba
    CKPT=/root/paddlejob/workspace/train_data/datasets/data107578/discoGAN5.9999.pdparams
    
    unzip -d $TO_DIR $CELEBA
    
    LOGDIR=/root/paddlejob/workspace/log/train_log
    OUTDIR=/root/paddlejob/workspace/output/model_imgs
    
    python -m paddle.distributed.launch --gpus '0,1,2,3' DiscoGAN-Paddle/discogan/image_translation.py \
        --image_dir $IMG_CELEBA \
        --attr_file $ATTR_TXT \
        --task_name celebA \
        --style_A Male \
        --n_test 200 \
        --batch_size 200 \
        --epoch_size 1000 \
        --result_path $OUTDIR \
        --model_path $OUTDIR \
        --log_out $LOGDIR \
        --local_rank 0 \
        --num_workers 4 \
        --learning_rate 0.0006 \
        --ckpt_path $CKPT \
        --iters 60000  # 预训练模型已经训练的迭代数
    ```

### Test

*现在训练了 gender 转换的模型*

- 设定数据集路径，加载预训练模型，设定输出路径:

```
python DiscoGAN-Paddle/discogan/evaluation.py \
  --image_dir path/to/img_align_celeba \
  --attr_file path/to/list_attr_celeba_demo.txt \
  --n_test 10 \
  --task_name celebA \
  --style_A Male \
  --batch_size 1 \
  --ckpt_path path/to/discoGAN.pdparams
  --result_path ./results
```

样例如下：

|A|AB|ABA|
| --- | --- | ---|
|![1A](./assets/figs/1.A.jpg)|![1AB](./assets/figs/1.AB.jpg)|![1ABA](./assets/figs/1.ABA.jpg)|
|![2B](./assets/figs/2.B.jpg)|![2BA](./assets/figs/2.BA.jpg)|![2BAB](./assets/figs/2.BAB.jpg)|
|![0B](./assets/figs/0.B.jpg)|![0BA](./assets/figs/0.BA.jpg)|![0BAB](./assets/figs/0.BAB.jpg)|
|![2A](./assets/figs/2.A.jpg)|![2AB](./assets/figs/2.AB.jpg)|![2ABA](./assets/figs/2.ABA.jpg)|

> 详见 `./assets/figs`

对于 CelebA 数据集，更多可以训练的跨域风格转换生成如论文中图 7 所示，训练方式见 `script.sh`。

![](./assets/conversions.jpg)


### 训练日志样例

```
11/03/2021 10:02:58 - image_translation.py[line:192] - INFO: num of iters: 422; len of test_loader: 1
11/03/2021 10:02:58 - image_translation.py[line:197] - INFO: [Process 6078] world_size = 1, rank = 0
11/03/2021 10:03:04 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 0 - Total GEN Loss: 1.0220423 - Total DIS Loss: 1.1977584 - GEN Loss: [1.0553604], [0.6670432] - Feature Matching Loss: [0.44880557], [0.5050385] - RECON Loss: [0.08394855], [0.08098113] - DIS Loss: [0.51311284], [0.6846456] 

11/03/2021 10:03:22 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 50 - Total GEN Loss: 0.3492102 - Total DIS Loss: 2.3001809 - GEN Loss: [1.1142987], [0.88896495] - Feature Matching Loss: [0.08826689], [0.07994663] - RECON Loss: [0.05301712], [0.04786431] - DIS Loss: [1.1721631], [1.1280177] 

11/03/2021 10:03:41 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 100 - Total GEN Loss: 0.2461912 - Total DIS Loss: 1.6372418 - GEN Loss: [0.8820798], [0.69587165] - Feature Matching Loss: [0.05643689], [0.04356069] - RECON Loss: [0.04652921], [0.0410934] - DIS Loss: [0.7744776], [0.86276424] 

11/03/2021 10:03:59 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 150 - Total GEN Loss: 0.1509379 - Total DIS Loss: 1.8644532 - GEN Loss: [0.44169796], [0.54509205] - Feature Matching Loss: [0.03311593], [0.02559727] - RECON Loss: [0.0480856], [0.04513963] - DIS Loss: [0.88916767], [0.97528553] 

11/03/2021 10:04:17 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 200 - Total GEN Loss: 0.1575147 - Total DIS Loss: 1.7311554 - GEN Loss: [0.65576416], [0.5651688] - Feature Matching Loss: [0.01968439], [0.02041043] - RECON Loss: [0.04742772], [0.04435579] - DIS Loss: [0.86194223], [0.8692131] 

11/03/2021 10:04:36 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 250 - Total GEN Loss: 0.1641732 - Total DIS Loss: 1.6536109 - GEN Loss: [0.74956435], [0.57868284] - Feature Matching Loss: [0.02182432], [0.01378206] - RECON Loss: [0.04651232], [0.04863647] - DIS Loss: [0.8441198], [0.80949116] 

11/03/2021 10:04:54 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 300 - Total GEN Loss: 0.167661 - Total DIS Loss: 1.6769731 - GEN Loss: [0.68102974], [0.4937142] - Feature Matching Loss: [0.034266], [0.02232504] - RECON Loss: [0.04847315], [0.04540079] - DIS Loss: [0.8367496], [0.84022355] 

11/03/2021 10:05:12 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 350 - Total GEN Loss: 0.1658249 - Total DIS Loss: 1.5696483 - GEN Loss: [0.71359026], [0.63911986] - Feature Matching Loss: [0.0186385], [0.01607903] - RECON Loss: [0.04882706], [0.04850165] - DIS Loss: [0.7888226], [0.7808256] 

11/03/2021 10:05:31 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 400 - Total GEN Loss: 0.1693972 - Total DIS Loss: 1.5416335 - GEN Loss: [0.6630931], [0.63824743] - Feature Matching Loss: [0.02085807], [0.02356325] - RECON Loss: [0.04797958], [0.05052868] - DIS Loss: [0.7721566], [0.7694769] 

11/03/2021 10:05:51 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 450 - Total GEN Loss: 0.1381095 - Total DIS Loss: 1.5566053 - GEN Loss: [0.6321756], [0.56732976] - Feature Matching Loss: [0.01026417], [0.01036626] - RECON Loss: [0.05042949], [0.04724529] - DIS Loss: [0.7780957], [0.7785096] 

11/03/2021 10:06:10 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 500 - Total GEN Loss: 0.1704209 - Total DIS Loss: 1.5026222 - GEN Loss: [0.63569707], [0.7344268] - Feature Matching Loss: [0.02235081], [0.01553744] - RECON Loss: [0.05420103], [0.04781702] - DIS Loss: [0.7489997], [0.75362253] 

11/03/2021 10:06:28 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 550 - Total GEN Loss: 0.1513395 - Total DIS Loss: 1.5154178 - GEN Loss: [0.6108958], [0.68345904] - Feature Matching Loss: [0.01308893], [0.01173348] - RECON Loss: [0.05455117], [0.05360683] - DIS Loss: [0.7572131], [0.7582047] 

11/03/2021 10:06:47 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 600 - Total GEN Loss: 0.1556971 - Total DIS Loss: 1.5222392 - GEN Loss: [0.6113478], [0.6610589] - Feature Matching Loss: [0.01180729], [0.02031383] - RECON Loss: [0.05379906], [0.05709589] - DIS Loss: [0.753572], [0.7686672] 

11/03/2021 10:07:05 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 650 - Total GEN Loss: 0.1544153 - Total DIS Loss: 1.5197338 - GEN Loss: [0.5871728], [0.73245037] - Feature Matching Loss: [0.01223763], [0.01323045] - RECON Loss: [0.05420906], [0.05384947] - DIS Loss: [0.76917875], [0.75055504] 

11/03/2021 10:07:23 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 700 - Total GEN Loss: 0.1573446 - Total DIS Loss: 1.477165 - GEN Loss: [0.6407921], [0.6757554] - Feature Matching Loss: [0.01509119], [0.0140161] - RECON Loss: [0.05470061], [0.05247597] - DIS Loss: [0.74690306], [0.7302619] 

11/03/2021 10:07:42 - image_translation.py[line:311] - INFO: Epoch: 1 - Iter: 750 - Total GEN Loss: 0.1555764 - Total DIS Loss: 1.4851844 - GEN Loss: [0.6116561], [0.7066419] - Feature Matching Loss: [0.01438331], [0.0126167] - RECON Loss: [0.04934706], [0.05144513] - DIS Loss: [0.74094343], [0.744241] 

11/03/2021 10:09:51 - image_translation.py[line:192] - INFO: num of iters: 422; len of test_loader: 1
11/03/2021 10:09:51 - image_translation.py[line:197] - INFO: [Process 6861] world_size = 1, rank = 0
11/03/2021 10:09:55 - image_translation.py[line:222] - INFO: resume ckpt from work/ckpts/discoGANhair.pdparams
11/03/2021 10:09:58 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 10000 - Total GEN Loss: 0.1692023 - Total DIS Loss: 1.360221 - GEN Loss: [0.8410817], [0.96437943] - Feature Matching Loss: [0.07857576], [0.01515051] - RECON Loss: [0.04227623], [0.03122869] - DIS Loss: [0.74404204], [0.616179] 

11/03/2021 10:10:16 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 10050 - Total GEN Loss: 0.1023466 - Total DIS Loss: 1.2999609 - GEN Loss: [0.78034663], [0.65781575] - Feature Matching Loss: [0.01887911], [0.0143446] - RECON Loss: [0.01827559], [0.01270013] - DIS Loss: [0.6561581], [0.64380276] 

11/03/2021 10:10:34 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 10100 - Total GEN Loss: 0.0979042 - Total DIS Loss: 1.2639713 - GEN Loss: [0.69233036], [0.72599435] - Feature Matching Loss: [0.01593208], [0.01511684] - RECON Loss: [0.01445001], [0.01158179] - DIS Loss: [0.6366608], [0.6273105] 

11/03/2021 10:10:52 - image_translation.py[line:311] - INFO: Epoch: 0 - Iter: 10150 - Total GEN Loss: 0.0935076 - Total DIS Loss: 1.2299426 - GEN Loss: [0.74897945], [0.55817586] - Feature Matching Loss: [0.01725106], [0.01744595] - RECON Loss: [0.01394007], [0.01113235] - DIS Loss: [0.60705733], [0.62288517] 

```

## 模型信息

| 信息 | 说明 |
| --- | --- |
| 发布者 | [刘辰](https://github.com/ttjygbtj)、[吴海涛](https://github.com/Dylan-get)、[石华榜](https://github.com/S-HuaBomb)、[杨瑞智](https://github.com/buriedms)、[许观](https://github.com/HeySUPERMELON) |
| 时间 | 2021.11 |
| 框架版本 | paddlepaddle==2.1.2 |
| 应用场景 | GAN 图像风格转换 |
| 支持硬件 | GPU × 4 |
| 预训练模型下载 | 链接：https://pan.baidu.com/s/1hB34lEKMdh4YIh84kQOILQ 提取码：sbsi |
| AI Studio 地址 | [DiscoGAN-Paddle](https://aistudio.baidu.com/aistudio/projectdetail/2548914?contributionType=1) |
