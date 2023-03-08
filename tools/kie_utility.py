import ast
from PIL import Image
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, str2bool, init_args as infer_args


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument("--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument("--table_char_dict_path", type=str, default="./ppocr/utils/dict/table_structure_dict_ch.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)
    parser.add_argument("--layout_dict_path",type=str, default="./ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
    parser.add_argument("--layout_score_threshold",type=float,default=0.5,help="Threshold of score.")
    parser.add_argument("--layout_nms_threshold",type=float,default=0.5, help="Threshold of nms.")
    # params for kie
    parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    parser.add_argument("--ser_model_dir", type=str, default='./saved_model/kie/')
    parser.add_argument("--ser_dict_path", type=str, default="./config/kie/labels_med.txt")
    # need to be None or tb-yx
    parser.add_argument("--ocr_order_method", type=str, default='tb-yx')
    # params for inference
    parser.add_argument("--mode", type=str, default='kie', help='structure and kie is supported')
    parser.add_argument("--image_orientation",type=bool, default=False, help='Whether to enable image orientation recognition')
    parser.add_argument("--layout", type=str2bool, default=True, help='Whether to enable layout analysis')
    parser.add_argument("--table", type=str2bool, default=True, help='In the forward, whether the table area uses table recognition')
    parser.add_argument("--ocr", type=str2bool, default=True, help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument("--recovery", type=str2bool, default=False, help='Whether to enable layout of recovery')
    parser.add_argument("--save_pdf", type=str2bool, default=False, help='Whether to save pdf file')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []
    for region in result:
        if region['type'] == 'table':
            pass
        else:
            for text_result in region['res']:
                boxes.append(np.array(text_result['text_region']))
                txts.append(text_result['text'])
                scores.append(text_result['confidence'])
    im_show = draw_ocr_box_txt(
        image, boxes, txts, scores, font_path=font_path, drop_score=0)
    return im_show
