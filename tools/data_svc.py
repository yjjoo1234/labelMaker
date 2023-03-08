import os  
import sys  
sys.path.insert(0, './')
from datetime import datetime  
import pymongo     
from bson.objectid import ObjectId  
from config.mongo import dbconnector 

def set_insertResult(seq_id, filename, images, origin_file, pg_count, data, vqa_data=None, user_id=''): 
    """
    의료비 처리 결과 저장 
    """ 

    # 'id': '', 
    # 'filename': '', 
    # 'images': [], 
    # 'origin_file': '', 
    # 'data': [], 
    # 'create_date': '', 
    # 'page_count': 0
    
    try:  
        db = dbconnector()  
        msg1 = {
            'id': seq_id,  
            'user_id': user_id, 
            'filename': filename,  
            'images': images,  
            'origin_file': origin_file, 
            'data': data, 
            'vqa_data': vqa_data, 
            'create_date': datetime.now(), 
            'page_count': pg_count  
        } 
        msg2 = {
            'id': seq_id,  
            'user_id': user_id, 
            'filename': filename,  
            'images': images,  
            'origin_file': origin_file, 
            'data': data,   
            'create_date': datetime.now(), 
            'page_count': pg_count  
        } 

        # 예측 결과 저장 (MongoDB)
        _id1 = db.med_expense.insert_one(msg1)
        # 평가 metrics용 데이터 (Ground Truth) 
        _id2 = db.med_expense_gt.insert_one(msg2)
           
        return str(_id1.inserted_id)   
    except Exception as ex:  
        return repr(ex)  


def get_medExpResult(doc_id): 
    """
    의료비 KIE 결과 리턴 
    """ 
    try: 
        result = {
            'response_code': 500, 
            'response_msg': '', 
            'id': '', 
            'filename': '', 
            'origin_file': '', 
            'images': [], 
            'data': [], 
            'create_date': '', 
            'page_count': 0
        } 
        db = dbconnector()        
        cur = db.med_expense.find_one({"id": str(doc_id)}) 
        if cur is None: 
            result['response_code'] = 500 
            result['response_msg'] = '해당 데이터가 없습니다.'
        else: 
            result['id'] = cur['id'] 
            result['filename'] = cur['filename']
            result['origin_file'] = cur['origin_file'] 
            result['images'] = cur['images']  
            result['data'] = cur['data']
            result['page_count'] = cur['page_count']
            result['create_date'] = cur['create_date'].strftime("%Y/%m/%d, %H:%M:%S")

            result['response_code'] = 200 
            result['response_msg'] = 'ok'  

        return result 

    except Exception as ex: 
        result['response_code'] = 500 
        result['response_msg'] = repr(ex)
        return result 