# FLAG3D++: A Benchmark for 3D Fitness Activity Comprehension with Language Instruction

The repository contains the official implementation of "FLAG3D++".

<div>
    <a href="https://andytang15.github.io/">
        Yansong Tang<sup>1</sup>
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://github.com/AY-Liu"/>
        Aoyang Liu<sup>1</sup>
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://moonsliu.github.io/">
        Jinpeng Liu<sup>1</sup>
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://shiyi-zh0408.github.io/">
        Shiyi Zhang<sup>1</sup>
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://github.com/Dai-Wenxun">
        Wenxun Dai<sup>1</sup>
    </a>&emsp;
    <!-- </br>Tsinghua University -->
    <a href="https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en">
        Jie Zhou<sup>1</sup>
    </a>&emsp;
    <a href="https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en">
        Xiu Li<sup>1</sup>
    </a>&emsp;
    <a href="https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN">
        Jiwen Lu<sup>1</sup>
    </a>
    <!-- </br>Tsinghua University -->
    </li>&emsp;
    <br>
    <sup>1</sup>Tsinghua University
</div>

![](images/teaser.png)

![](images/figure1.png)

![](images/figure2.png)

## Set up

### Environment

```bash
conda create -n hl-rac python==3.8
```

We recommend installing pytorch 1.14 to run the code.

```bash
pip install -r requirements.txt
```

### Data

The whole data and corresponding weights can be obtained from this [website](https://cloud.tsinghua.edu.cn/d/c742cc573d32460f9af4/). To make it easier to reproduce the results as well as to use them, you can directly use the already processed file (processed_data). If you have other scientific needs, you can also use raw files (raw_data).

### Test

Please run following command:

```bash
cd hl-rac
python test.py
```

## Acknowledgement

Our code is based on [FLAG3D](https://andytang15.github.io/FLAG3D/), [2s-AGCN](https://github.com/lshiwjx/2s-AGCN), [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC), thanks to all the contributors!
