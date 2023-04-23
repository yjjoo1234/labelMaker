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



## 2. Explanation

### 2.1 단축키

| 단축키                     | 설명                                              |
|--------------------------|--------------------------------------------------|
| Ctrl + Shift + R         | 현재이미지 텍스트 인식 (Text Recognition.)             |
| W                        | BBox 생성                                          |
| Q                        | 멀티 포인트 박스 생성                                  |
| X                        | 반시계 방향으로 이미지 회전                              |
| C                        | 시계방향으로 이미지 회전                                |
| Ctrl + E                 | 선택 박스 라벨 수정                                   |
| Ctrl + X                 | 박스 Key 변경 (Key Information Extraction)          |
| Ctrl + R                 | 선택 박스 재인식 (Text Re-recoginition)               |
| Ctrl + C                 | 선택박스 복사/붙여넣기                                  |
| Ctrl + Left Mouse Button | 여러 라벨 박스 선택                                    |
| Backspace                | 선택 박스 삭제 처리                                    |
| Ctrl + V                 | 이미지 체크(라벨로 저장)                                |
| Ctrl + Shift + d         | 이미지 삭제                                          |
| D                        | 다음 이미지                                          |
| A                        | 이전 이미지                                          |
| Ctrl++                   | Zoom in                                           |
| Ctrl--                   | Zoom out                                          |
| ↑→↓←                     | 선택박스 이동                                        |

### 2.2 Built-in Model



# labelMaker
