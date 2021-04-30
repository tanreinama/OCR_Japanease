import numpy as np
import cv2
import gc
import os
import json
import argparse
import torch
from nets.detectionnet import get_detectionnet
from nets.classifiernet import get_classifiernet
from misc.nihongo import nihongo_class, filter_word
from misc.detection import Detector
from misc.nms import non_max_suppression, column_wordlines

parser = argparse.ArgumentParser()
parser.add_argument('images', metavar='file', type=str, nargs='+',
                    help='input image files')
parser.add_argument("--dpi", type=int, default=-1, help="image dpi")
parser.add_argument('--cpu', action='store_true', help="CPU mode (no GPU)")
parser.add_argument('--output_format', type=str, default="row", help="output format", choices=['row', 'json'])
parser.add_argument('--output_detect_img', action='store_true', help="output detected bounding box")
parser.add_argument('--low_gpu_memory', action='store_true', help="reduce gpu memory usage")
args = parser.parse_args()

def main():
    d = []
    for f in args.images:
        if os.path.isfile(f):
            d.append(f)
        elif os.path.isdir(f):
            d.extend([f+'/'+a for a in os.listdir(f)])
        else:
            print('Input file "%s" in not file or directory.'%file)
            return
    ocr_result = get_ocr(d, dpi=args.dpi, use_cuda=(not args.cpu), output_detect_img=args.output_detect_img, low_gpu_memory=args.low_gpu_memory)
    if args.output_format == 'json':
        print(json.dumps(ocr_result, ensure_ascii=False))
    else:
        for r in ocr_result:
            print('file "%s" detected in %d dpi.'%(r['filename'],r['detected_dpi']))
            for b in r['blocks']:
                print('[Block #%d]'%b['id'])
                for s in b['sentences']:
                    print(s['sent'])

def filter_block(sent):
    for i in range(len(sent)):
        for j in range(len(filter_word)):
            if filter_word[j][0] == sent[i]:
                if filter_word[j][2] == "":
                    bef = (i==0)
                else:
                    bef = filter_word[j][2] is None or (i>0 and sent[i-1] in filter_word[j][2])
                if filter_word[j][3] == "":
                    aft = (i==len(sent)-1)
                else:
                    aft = filter_word[j][3] is None or (i<len(sent)-1 and sent[i+1] in filter_word[j][3])
                if bef and aft:
                    sent[i] = filter_word[j][1]

def get_ocr(filelist,dpi,use_cuda=True,output_detect_img=False,low_gpu_memory=False):
    det_model = 'models/detectionnet.model'
    cls_model = 'models/classifiernet.model'
    if not (os.path.isfile(det_model) and os.path.isfile(cls_model)):
        print('Model file not found.')
        return
    for file in filelist:
        if not os.path.isfile(file):
            print('Input file "%s" not found.'%file)
            return
    model = get_detectionnet()
    if use_cuda:
        model.load_state_dict(torch.load(det_model))
    else:
        model.load_state_dict(torch.load(det_model, map_location=torch.device('cpu')))
    dt = Detector(use_cuda=use_cuda, low_gpu_memory=low_gpu_memory)
    detections = []
    for file in filelist:
        im = cv2.imread(file)
        if im is None:
            print('Cannot read input file "%s".'%file)
            return
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        elif len(im.shape) != 2:
            print('Cannot read input file "%s".'%file)
            return
        d = dt.detect_image(model, im, dpi)
        detections.append(d)
    del model
    torch.cuda.empty_cache()
    model = get_classifiernet(len(nihongo_class) + 2)
    if use_cuda:
        model.load_state_dict(torch.load(cls_model))
    else:
        model.load_state_dict(torch.load(cls_model, map_location=torch.device('cpu')))
    boundings = []
    for d in detections:
        b = dt.bounding_box(model, d)
        b = non_max_suppression(b)
        boundings.append(b)
    del model
    torch.cuda.empty_cache()
    results = []
    for file, dtct, bbox in zip(filelist, detections, boundings):
        detected_dpi, _, gray_img, scale_image, _ = dtct
        detect_file = {'filename':file,'detected_dpi':detected_dpi,'blocks':[]}
        if output_detect_img:
            detect_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            detect_img = cv2.cvtColor(detect_img, cv2.COLOR_GRAY2RGB)
        if len(bbox) > 0:
            for i in range(max([b.sentenceindex for b in bbox])+1):
                bbox_sent = [b for b in bbox if b.sentenceindex == i]
                cbox = column_wordlines(bbox_sent)
                block_one = {'id':i,'sentences':[]}
                for c in cbox:
                    blk = []
                    box = []
                    for b in c:
                        n, s = b.word()
                        if s > 0.3:
                            x1, y1, x2, y2 = b.x1 * 2, b.y1 * 2, b.x2 * 2, b.y2 * 2
                            x1 = int(np.round(x1 * scale_image[0]))
                            y1 = int(np.round(y1 * scale_image[1]))
                            x2 = int(np.round(x2 * scale_image[0]))
                            y2 = int(np.round(y2 * scale_image[1]))
                            x1 = min(max(0, x1), gray_img.shape[1])
                            y1 = min(max(0, y1), gray_img.shape[0])
                            x2 = min(max(0, x2), gray_img.shape[1])
                            y2 = min(max(0, y2), gray_img.shape[0])
                            blk.append(nihongo_class[n])
                            box.append((x1, y1, x2, y2, float(s)))
                            if output_detect_img:
                                cv2.rectangle(detect_img, (x1,y1), (x2,y2), (255,0,0), 2)
                    filter_block(blk)
                    sent = ''.join(blk)
                    sent_one = {'sent':sent,'bbox':[]}
                    for w,b in zip(blk,box):
                        box_one = {'word':w,'box':[b[0],b[1],b[2],b[3]],'score':b[4]}
                        sent_one['bbox'].append(box_one)
                    block_one['sentences'].append(sent_one)
                detect_file['blocks'].append(block_one)
                if output_detect_img:
                    bb = [b for s in block_one['sentences'] for b in s['bbox']]
                    if len(bb) > 0:
                        x1 = min([b['box'][0] for b in bb])
                        y1 = min([b['box'][1] for b in bb])
                        x2 = max([b['box'][2] for b in bb])
                        y2 = max([b['box'][3] for b in bb])
                        cv2.rectangle(detect_img, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(detect_img, '#%d'%block_one['id'], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if output_detect_img:
            cv2.imwrite(file+'-detections.png', detect_img)
        results.append(detect_file)
    return results

if __name__ == '__main__':
    main()
