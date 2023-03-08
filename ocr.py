import os
import sys
import importlib
__dir__ = os.path.dirname(__file__) 
sys.path.append(os.path.join(__dir__, '')) 
import cv2
import logging
import numpy as np 
tools = importlib.import_module('.', 'tools')
ppocr = importlib.import_module('.', 'ppocr')
from tools.infer import predict_system
from tools.kie_utility import init_args  
from tools.infer.utility import draw_ocr, str2bool, check_gpu 
from ppocr.utils.logging import get_logger 
logger = get_logger()
from ppocr.utils.utility import check_and_read 
from ppocr.utils.network import download_with_progressbar  


__all__ = ['PaddleOCR', 'draw_ocr' ]

VERSION = '2.6.1.0'
DEFAULT_OCR_MODEL_VERSION = 'OCRv3'
SUPPORT_OCR_MODEL_VERSION = ['OCR', 'OCRv2', 'OCRv3'] 
SUPPORT_DET_MODEL = ['DB', 'DB++']
SUPPORT_REC_MODEL = ['CRNN', 'SVTR', 'SVTR_LCNet', 'ABINet'] 

def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='korean')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument("--ocr_version", type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='OCRv3',
        help='OCR Model version' 
    )
    parser.add_argument("--structure_version", type=str, 
        default='STRUCTURE',
        help='Structure Model version')
 
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)
  

class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """ 
        args: 
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = True #check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls 

        # init model dir 
        if params.lang == 'ko':    
            params.det_model_dir = './saved_model/det'         
            params.rec_model_dir = './saved_model/rec' 
            params.rec_char_dict_path = './config/rec/korean_dict.txt'  
            params.cls_model_dir = './saved_model/cls'  
        elif params.lang == 'en': 
            params.det_model_dir = './saved_model/det_en'         
            params.rec_model_dir = './saved_model/rec_en' 
            params.rec_char_dict_path = './config/rec/en_dict.txt'  
            params.cls_model_dir = './saved_model/cls'   
        else: 
            params.det_model_dir = './saved_model/det_zh'         
            params.rec_model_dir = './saved_model/rec_zh' 
            params.rec_char_dict_path = './config/rec/ppocr_keys_v1.txt' 
            params.cls_model_dir = './saved_model/cls'  

        if params.ocr_version == 'OCRv3':
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320" 

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0) 
            
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=False):
        """
        ocr model 
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. 
                If true, the text with rotation of 180 degrees can be recognized.
                If no text is rotated by 180 degrees, use cls=False to get better performance. 
                Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
            )

        if isinstance(img, str):
            # download net image
            if img.startswith('http'):
                download_with_progressbar(img, 'tmp.jpg')
                img = 'tmp.jpg'
            image_file = img
            img, flag, _ = check_and_read(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if det and rec:
            dt_boxes, rec_res, time_dict = self.__call__(img, cls)
            print(time_dict)
            return [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            if dt_boxes is None:
                return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec:
                    return cls_res
            rec_res, elapse = self.text_recognizer(img)
            return rec_res

 