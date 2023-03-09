# labelMaker (based on PPOCRLabelv2)

labelMaker는 한글 OCR 학습용 데이터(Text Detection & Recognition, Key Information Extraction) 생성을 위한 툴입니다. 
 

## 1. Installation and Run

### 1.1 Install PaddlePaddle 

```bash
pip3 install --upgrade pip

# gpu version: cuda9 or cuda10 
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# cpu version 
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
``` 
gpu version: cuda11+  
[visit PaddlePaddle site](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/macos-pip_en.html)  
 
For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.

### 1.2 Run labelMaker

```bash 
python labelMaker.py 

``` 

#### Windows

```bash 
 
python labelMaker.py  
```

> If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows. Please try to download Shapely whl file using http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely.
>
> Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)
>

#### Linux

```bash 
 
python labelMaker.py  
```

#### MacOS
```bash 
pip3 install opencv-contrib-python-headless==4.2.0.32

python labelMaker.py # [KIE mode] for [detection + recognition + keyword extraction] labeling
```

#### 1.2.2 Run labelMaker by Python Script 

```bash
cd ./labelMaker  # Switch to the labelMaker directory

# Select label mode and run 
python labelMaker.py   
```
 

## 2. Usage

### 2.1 Steps
 
 
### 2.2 Note
 

## 3. Explanation

### 3.1 단축키 

| 단축키                     | 설명                                              |
|--------------------------|--------------------------------------------------|
| Ctrl + Shift + R         | Re-recognize all the labels of the current image |
| W                        | Create a rect box                                |
| Q                        | Create a multi-points box                         |
| X                        | Rotate the box anti-clockwise                    |
| C                        | Rotate the box clockwise                         |
| Ctrl + E                 | Edit label of the selected box                   |
| Ctrl + X                 | Change key class of the box when enable `--kie`  |
| Ctrl + R                 | Re-recognize the selected box                    |
| Ctrl + C                 | Copy and paste the selected box                  |
| Ctrl + Left Mouse Button | Multi select the label box                       |
| Backspace                | Delete the selected box                          |
| Ctrl + V                 | Check image                                      |
| Ctrl + Shift + d         | Delete image                                     |
| D                        | Next image                                       |
| A                        | Previous image                                   |
| Ctrl++                   | Zoom in                                          |
| Ctrl--                   | Zoom out                                         |
| ↑→↓←                     | Move selected box                                |

### 3.2 Built-in Model 

### 3.4 Export Partial Recognition Results
 
### 3.5 Dataset division
 
  
### 3.6 Error message
 

### 4. Related
 
