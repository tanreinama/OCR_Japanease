import numpy as np
import cv2
import torch
import math
from sklearn.cluster import OPTICS
from .nihongo import nihongo_class

class BoundingBox(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.detectionscore = 0.0
        self.sentenceindex = 0
        self.prediction = None
    def set_prediction(self, pred):
        self.prediction = pred
    def iswordbox(self):
        return self.prediction is not None and np.argmax(self.prediction) < len(nihongo_class)
    def isword(self):
        return self.iswordbox() and np.argmax(self.prediction) > 0
    def word(self):
        r = [(a, self.prediction[a]) for a in range(len(self.prediction)) if a > 0 and a < len(nihongo_class)]
        return sorted(r, key=lambda x:x[1])[-1]
    def score(self):
        _, classifiercore = self.word()
        return 5 * self.detectionscore + classifiercore

class BoundingBoxDataset(object):
    def __init__(self, org_img, scale_wh, boundbox, output_size=(40,40)):
        self.org_img = org_img
        self.scale_wh = scale_wh
        self.boundbox = boundbox
        self.output_size = output_size
    def __getitem__(self, idx):
        bb = self.boundbox[idx]
        x1, y1, x2, y2 = bb.x1 * 4, bb.y1 * 4, bb.x2 * 4, bb.y2 * 4
        x1 = int(np.round(x1 * self.scale_wh[0]))
        y1 = int(np.round(y1 * self.scale_wh[1]))
        x2 = int(np.round(x2 * self.scale_wh[0]))
        y2 = int(np.round(y2 * self.scale_wh[1]))
        x1 = min(max(0, x1), self.org_img.shape[1])
        y1 = min(max(0, y1), self.org_img.shape[0])
        x2 = min(max(0, x2), self.org_img.shape[1])
        y2 = min(max(0, y2), self.org_img.shape[0])
        im = self.org_img[y1:y2,x1:x2]
        im = cv2.resize(im, self.output_size)
        im = im.reshape((1,self.output_size[1],self.output_size[0]))
        return im.astype(np.float32) / 255.
    def __len__(self):
        return len(self.boundbox)

class SentenceBox(object):
    def __init__(self, word_threshold):
        self.boundingboxs = []
        self.word_threshold = word_threshold

    def _conv3_filter(self, img, pos, pos_div):
        points = []
        rects = []
        for y in range(1,img.shape[0]-1):
            for x in range(1,img.shape[1]-1):
                if img[y,x]>self.word_threshold and img[y,x]>img[y-1,x] and img[y,x]>img[y-1,x-1] and img[y,x]>img[y-1,x+1] and img[y,x]>img[y,x-1] and img[y,x]>img[y,x+1] and img[y,x]>img[y+1,x] and img[y,x]>img[y+1,x-1] and img[y,x]>img[y+1,x+1]:
                    points.append((x,y))
                    w, h = pos[0][y//pos_div,x//pos_div]/2, pos[1][y//pos_div,x//pos_div]/2
                    w, h = 1+w*img.shape[1], 1+h*img.shape[0]
                    offw, offh = int(np.round(w)), int(np.round(h))
                    rects.append((x-offw,y-offh,x+offw,y+offh))
        return points, rects

    def make_boundingbox(self, hm_wd, hm_pos, pos_div=2, min_bound=24, resize_val=1.1, aspect_val=1.25):
        pos, rcts = self._conv3_filter(hm_wd, hm_pos, pos_div)
        min_bound = min_bound // 4
        for p, r in zip(pos, rcts):
            x1, y1, x2, y2 = r
            x1 = min(max(0,x1), hm_wd.shape[1])
            y1 = min(max(0,y1), hm_wd.shape[0])
            x2 = min(max(0,x2), hm_wd.shape[1])
            y2 = min(max(0,y2), hm_wd.shape[0])
            w, h = x2-x1, y2-y1
            if min(w, h) >= min_bound:
                self.boundingboxs.append(BoundingBox(x1, y1, x2, y2))
            w2 = int(np.round((x2-x1) * resize_val))
            h2 = int(np.round((y2-y1) * resize_val))
            if w2 != w and h2 != h and min(w2, h2) >= min_bound:
                xx1 = min(max(0, p[0] - w2//2), hm_wd.shape[1])
                yy1 = min(max(0, p[1] - h2//2), hm_wd.shape[0])
                xx2 = min(max(0, p[0] - w2//2 + w2), hm_wd.shape[1])
                yy2 = min(max(0, p[1] - h2//2 + h2), hm_wd.shape[0])
                self.boundingboxs.append(BoundingBox(xx1, yy1, xx2, yy2))
            w2 = int(np.round((x2-x1) * aspect_val))
            h2 = int(np.round((y2-y1) / aspect_val))
            if w2 != w and h2 != h and min(w2, h2) >= min_bound:
                xx1 = min(max(0, p[0] - w2//2), hm_wd.shape[1])
                yy1 = min(max(0, p[1] - h2//2), hm_wd.shape[0])
                xx2 = min(max(0, p[0] - w2//2 + w2), hm_wd.shape[1])
                yy2 = min(max(0, p[1] - h2//2 + h2), hm_wd.shape[0])
                self.boundingboxs.append(BoundingBox(xx1, yy1, xx2, yy2))
            w2 = int(np.round((x2-x1) / aspect_val))
            h2 = int(np.round((y2-y1) * aspect_val))
            if w2 != w and h2 != h and min(w2, h2) >= min_bound:
                xx1 = min(max(0, p[0] - w2//2), hm_wd.shape[1])
                yy1 = min(max(0, p[1] - h2//2), hm_wd.shape[0])
                xx2 = min(max(0, p[0] - w2//2 + w2), hm_wd.shape[1])
                yy2 = min(max(0, p[1] - h2//2 + h2), hm_wd.shape[0])
                self.boundingboxs.append(BoundingBox(xx1, yy1, xx2, yy2))

    def make_detectionscore(self, hm_wd_all):
        for i in range(len(self.boundingboxs)):
            x1, y1, x2, y2 = self.boundingboxs[i].x1, self.boundingboxs[i].y1, self.boundingboxs[i].x2, self.boundingboxs[i].y2
            y_pred = hm_wd_all[y1:y2,x1:x2]
            w, h = x2-x1, y2-y1
            y_true = ((np.exp(-(((np.arange(w)-(w/2))/(w/10))**2)/2)).reshape(1,-1)
                       *(np.exp(-(((np.arange(h)-(h/2))/(h/10))**2)/2)).reshape(-1,1))
            self.boundingboxs[i].detectionscore = 1.0 - np.mean((y_pred-y_true)**2)

    def make_sentenceid(self, id):
        for i in range(len(self.boundingboxs)):
            self.boundingboxs[i].sentenceindex = id

class Detector(object):
    def __init__(self, use_cuda=True, sentence_threshold=0.007, word_threshold=0.01):
        self.use_cuda = use_cuda
        self.sentence_threshold = sentence_threshold
        self.word_threshold = word_threshold

    def _get_class(self, im):
        minmax = (im.min(), im.max())
        if minmax[1]-minmax[0] == 0:
            return np.array()
        im = (im-minmax[0]) / (minmax[1]-minmax[0])
        clf = OPTICS(metric='euclidean', min_cluster_size=75)
        a = []
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if im[x][y] > self.sentence_threshold:
                    a.append([x,y])
        b = clf.fit_predict(a)
        c = np.zeros(im.shape)
        for i in range(len(b)):
            c[a[i][0],a[i][1]] = b[i]+1
        return c

    def _get_map(self, clz_map):
        all_map = []
        for i in range(1,int(np.max(clz_map)+1)):
            clz_wd = np.zeros(clz_map.shape, dtype=np.uint8)
            where = np.where(clz_map == i)
            clz_wd[where] = 255
            cnts = cv2.findContours(clz_wd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).reshape((-1,1,2)).astype(np.int32)
                cv2.drawContours(clz_wd,[box],0,255,2)
                cv2.drawContours(clz_wd,[box],0,255,-1)
            all_map.append(clz_wd)
        return all_map

    def _filt_map(self, all_map):
        maps = []
        dindx = []
        for i1, m1 in enumerate(all_map):
            for i2, m2 in enumerate(all_map):
                if i1 != i2:
                    if np.sum(m2[m1 != 0]) == np.sum(m2):
                        dindx.append(i2)
        for i1, m1 in enumerate(all_map):
            if not i1 in dindx:
                for i2, m2 in enumerate(all_map[i1+1:]):
                    an = ((m1 == 0) + (m2 == 0)) == 0
                    if np.sum(an) != 0:
                        if np.sum(m1) > np.sum(m2):
                            m2[an] = 0
                        else:
                            m1[an] = 0
                maps.append(m1)
        return np.array(maps)

    def _scale_image(self, img, long_size):
        if img.shape[0] < img.shape[1]:
            scale = img.shape[1] / long_size
            size = (long_size, math.ceil(img.shape[0] / scale))
        else:
            scale = img.shape[0] / long_size
            size = (math.ceil(img.shape[1] / scale), long_size)
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    def _detect1x(self, detector_model, gray_img_scaled):
        im = np.zeros((1,1,512,512))
        im[0,0,0:gray_img_scaled.shape[0],0:gray_img_scaled.shape[1]] = gray_img_scaled
        x = np.clip(im / 255, 0.0, 1.0).astype(np.float32)
        x = torch.tensor(x)
        dp = torch.nn.DataParallel(detector_model)
        if self.use_cuda:
            x = x.cuda()
            dp = dp.cuda()
        dp.eval()
        y = dp(x)

        hm_wd = ((y[0]['hm_wd'] + y[1]['hm_wd']) / 2).detach().cpu().numpy().reshape(128,128)
        hm_sent = ((y[0]['hm_sent'] + y[1]['hm_sent']) / 2).detach().cpu().numpy().reshape(128,128)
        hm_pos = ((y[0]['of_size'] + y[1]['of_size']) / 2).detach().cpu().numpy().reshape(2,64,64)
        del x, y
        if self.use_cuda:
            torch.cuda.empty_cache()
        return hm_wd, hm_sent, hm_pos

    def _detect4x(self, detector_model, gray_img_scaled):
        tmp = np.zeros((1024,1024))
        tmp[0:gray_img_scaled.shape[0],0:gray_img_scaled.shape[1]] = gray_img_scaled
        im = np.zeros((4,1,512,512))
        im[0,0] = tmp[0:512,0:512]
        im[1,0] = tmp[512:1024,0:512]
        im[2,0] = tmp[0:512,512:1024]
        im[3,0] = tmp[512:1024,512:1024]
        x = np.clip(im / 255, 0.0, 1.0).astype(np.float32)
        x = torch.tensor(x)
        dp = torch.nn.DataParallel(detector_model)
        if self.use_cuda:
            x = x.cuda()
            dp = dp.cuda()
        dp.eval()
        y = dp(x)

        hm_wd = np.zeros((256,256))
        hm_sent = np.zeros((256,256))
        hm_pos = np.zeros((2,128,128))
        hm_wd[0:128,0:128] = ((y[0]['hm_wd'][0] + y[1]['hm_wd'][0]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_wd[128:256,0:128] = ((y[0]['hm_wd'][1] + y[1]['hm_wd'][1]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_wd[0:128,128:256] = ((y[0]['hm_wd'][2] + y[1]['hm_wd'][2]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_wd[128:256,128:256] = ((y[0]['hm_wd'][3] + y[1]['hm_wd'][3]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_sent[0:128,0:128] = ((y[0]['hm_sent'][0] + y[1]['hm_sent'][0]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_sent[128:256,0:128] = ((y[0]['hm_sent'][1] + y[1]['hm_sent'][1]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_sent[0:128,128:256] = ((y[0]['hm_sent'][2] + y[1]['hm_sent'][2]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_sent[128:256,128:256] = ((y[0]['hm_sent'][3] + y[1]['hm_sent'][3]) / 2).detach().cpu().numpy().reshape(128,128)
        hm_pos[:,0:64,0:64] = ((y[0]['of_size'][0] + y[1]['of_size'][0]) / 2 / 2).detach().cpu().numpy().reshape(2,64,64)
        hm_pos[:,64:128,0:64] = ((y[0]['of_size'][1] + y[1]['of_size'][1]) / 2 / 2).detach().cpu().numpy().reshape(2,64,64)
        hm_pos[:,0:64,64:128] = ((y[0]['of_size'][2] + y[1]['of_size'][2]) / 2 / 2).detach().cpu().numpy().reshape(2,64,64)
        hm_pos[:,64:128,64:128] = ((y[0]['of_size'][3] + y[1]['of_size'][3]) / 2 / 2).detach().cpu().numpy().reshape(2,64,64)
        del x, y
        if self.use_cuda:
            torch.cuda.empty_cache()
        return hm_wd, hm_sent, hm_pos

    def _detect16x(self, detector_model, gray_img_scaled):
        tmp = np.zeros((2048,2048))
        tmp[0:gray_img_scaled.shape[0],0:gray_img_scaled.shape[1]] = gray_img_scaled
        hm_wd = np.zeros((512,512))
        hm_sent = np.zeros((512,512))
        hm_pos = np.zeros((2,256,256))
        dp = torch.nn.DataParallel(detector_model)
        if self.use_cuda:
            dp = dp.cuda()
        dp.eval()
        for ygrid_i in range(4):
            im = np.zeros((4,1,512,512))
            im[0,0] = tmp[512*ygrid_i:512*ygrid_i+512,0:512]
            im[1,0] = tmp[512*ygrid_i:512*ygrid_i+512,512:1024]
            im[2,0] = tmp[512*ygrid_i:512*ygrid_i+512,1024:1536]
            im[3,0] = tmp[512*ygrid_i:512*ygrid_i+512,1536:2048]
            x = np.clip(im / 255, 0.0, 1.0).astype(np.float32)
            x = torch.tensor(x)
            if self.use_cuda:
                x = x.cuda()
            y = dp(x)

            hm_wd[128*ygrid_i:128*ygrid_i+128,0:128] = ((y[0]['hm_wd'][0] + y[1]['hm_wd'][0]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_wd[128*ygrid_i:128*ygrid_i+128,128:256] = ((y[0]['hm_wd'][1] + y[1]['hm_wd'][1]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_wd[128*ygrid_i:128*ygrid_i+128,256:384] = ((y[0]['hm_wd'][2] + y[1]['hm_wd'][2]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_wd[128*ygrid_i:128*ygrid_i+128,384:512] = ((y[0]['hm_wd'][3] + y[1]['hm_wd'][3]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_sent[128*ygrid_i:128*ygrid_i+128,0:128] = ((y[0]['hm_sent'][0] + y[1]['hm_sent'][0]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_sent[128*ygrid_i:128*ygrid_i+128,128:256] = ((y[0]['hm_sent'][1] + y[1]['hm_sent'][1]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_sent[128*ygrid_i:128*ygrid_i+128,256:384] = ((y[0]['hm_sent'][2] + y[1]['hm_sent'][2]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_sent[128*ygrid_i:128*ygrid_i+128,384:512] = ((y[0]['hm_sent'][3] + y[1]['hm_sent'][3]) / 2).detach().cpu().numpy().reshape(128,128)
            hm_pos[:,64*ygrid_i:64*ygrid_i+64,0:64] = ((y[0]['of_size'][0] + y[1]['of_size'][0]) / 2 / 4).detach().cpu().numpy().reshape(2,64,64)
            hm_pos[:,64*ygrid_i:64*ygrid_i+64,64:128] = ((y[0]['of_size'][1] + y[1]['of_size'][1]) / 2 / 4).detach().cpu().numpy().reshape(2,64,64)
            hm_pos[:,64*ygrid_i:64*ygrid_i+64,128:192] = ((y[0]['of_size'][2] + y[1]['of_size'][2]) / 2 / 4).detach().cpu().numpy().reshape(2,64,64)
            hm_pos[:,64*ygrid_i:64*ygrid_i+64,192:256] = ((y[0]['of_size'][3] + y[1]['of_size'][3]) / 2 / 4).detach().cpu().numpy().reshape(2,64,64)
            del x, y
            if self.use_cuda:
                torch.cuda.empty_cache()
        return hm_wd, hm_sent, hm_pos

    def _get_maps(self, detector_model, gray_img, dpi, min_word_size_cm):
        long_size = max(gray_img.shape)
        inch_size = long_size / dpi
        pix_size = int(np.round(52 * (inch_size * 2.54)))
        if pix_size <= 512:
            detect_size = 512
        elif pix_size <= 1024:
            detect_size = 1024
        else:
            detect_size = 2048

        div_size = min(pix_size,detect_size)
        pix_image = self._scale_image(gray_img, div_size)
        gray_img_scaled = np.zeros((detect_size, detect_size), dtype=np.uint8)
        gray_img_scaled[0:pix_image.shape[0],0:pix_image.shape[1]] = pix_image
        scale_image = (gray_img.shape[1]/pix_image.shape[1], gray_img.shape[0]/pix_image.shape[0])

        if detect_size == 512:
            hm_wd, hm_sent, hm_pos = self._detect1x(detector_model, gray_img_scaled)
        elif detect_size == 1024:
            hm_wd, hm_sent, hm_pos = self._detect4x(detector_model, gray_img_scaled)
        elif detect_size == 2048:
            hm_wd, hm_sent, hm_pos = self._detect16x(detector_model, gray_img_scaled)
        return pix_image, scale_image, hm_wd, hm_sent, hm_pos

    def _find_best_dpi(self, detector_model, gray_img, dpi, min_word_size_cm):
        for testdpi in (72,100,150,200,300):
            res = self._get_maps(detector_model, gray_img, testdpi, min_word_size_cm)
            if np.max(res[3]) > 0.5:
                return testdpi, res
        return 300, res

    def detect_image(self, detector_model, gray_img, dpi=72, min_word_size_cm=0.5):
        min_bound = int(np.round(min_word_size_cm * dpi / 2.54))

        if dpi > 0:
            pix_image, scale_image, hm_wd, hm_sent, hm_pos = self._get_maps(detector_model, gray_img, dpi, min_word_size_cm)
        else:
            dpi, (pix_image, scale_image, hm_wd, hm_sent, hm_pos) = self._find_best_dpi(detector_model, gray_img, dpi, min_word_size_cm)

        class_map = self._get_class(hm_sent)
        all_map = self._get_map(class_map)
        all_map = self._filt_map(all_map)

        sent_box = []

        pos_div = 2 if detector_model.use_offset_pooling else 1
        for i, now_map in enumerate(all_map):
            clz_wd = hm_wd.copy()
            clz_wd[now_map == 0] = 0
            clz_wd = (clz_wd - np.min(clz_wd)) / (np.max(clz_wd) - np.min(clz_wd))
            sbox = SentenceBox(self.word_threshold)
            sbox.make_boundingbox(clz_wd, hm_pos, pos_div, min_bound)
            if len(sbox.boundingboxs) > 0:
                sent_box.append(sbox)

        return dpi, sent_box, gray_img, scale_image, hm_wd

    def _bounding_box(self, classifier_model, gray_img, scale_image, boundings, batch_size_classifier, num_workers):
        dataset = BoundingBoxDataset(gray_img, scale_image, boundings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_classifier, shuffle=False, num_workers=num_workers)
        dp = torch.nn.DataParallel(classifier_model)
        if self.use_cuda:
            dp = dp.cuda()
        dp.eval()

        num_pred = 0
        for x in loader:
            if self.use_cuda:
                x = x.cuda()
            val = dp(x)
            val = torch.nn.functional.softmax(val, dim=1)
            val = val.detach().cpu().numpy()
            for v in val:
                boundings[num_pred].set_prediction(v)
                num_pred += 1
            del x
        del dp
        if self.use_cuda:
            torch.cuda.empty_cache()

    def bounding_box(self, classifier_model, detection, batch_size_classifier=32, num_workers=2, repeat_box=1):
        dpi, sent_box, gray_img, scale_image, hm_wd = detection

        for i, sbox in enumerate(sent_box):
            sbox.make_sentenceid(i)
            sbox.make_detectionscore(hm_wd)

        all_bounding = sum([sbox.boundingboxs for sbox in sent_box], [])
        ignore_idx = []
        self._bounding_box(classifier_model, gray_img, scale_image, all_bounding, batch_size_classifier, num_workers)

        for _ in range(repeat_box):
            extra_bounding = []
            for i, bbox in enumerate(all_bounding):
                if (not bbox.iswordbox()) and (i not in ignore_idx):
                    ignore_idx.append(i)
                    classindex = np.argmax(bbox.prediction)
                    if classindex == len(nihongo_class):  # 横並び
                        w, h = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
                        cx = (bbox.x1 + bbox.x2) // 2
                        if cx-w > 0:
                            b1 = BoundingBox(cx-w, bbox.y1, cx, bbox.y2)
                            b1.sentenceindex = bbox.sentenceindex
                            extra_bounding.append(b1)
                        if cx+w < hm_wd.shape[1]:
                            b2 = BoundingBox(cx, bbox.y1, cx+w, bbox.y2)
                            b2.sentenceindex = bbox.sentenceindex
                            extra_bounding.append(b2)
                    elif classindex == len(nihongo_class)+1:  # 縦並び
                        w, h = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
                        cy = (bbox.y1 + bbox.y2) // 2
                        if cy-h > 0:
                            b1 = BoundingBox(bbox.x1, cy-h, bbox.x2, cy)
                            b1.sentenceindex = bbox.sentenceindex
                            extra_bounding.append(b1)
                        if cy+h < hm_wd.shape[0]:
                            b2 = BoundingBox(bbox.x1, cy, bbox.x2, cy+h)
                            b2.sentenceindex = bbox.sentenceindex
                            extra_bounding.append(b2)

            if len(extra_bounding) == 0:
                break

            sbox = SentenceBox(self.word_threshold)
            sbox.boundingboxs = extra_bounding
            sbox.make_detectionscore(hm_wd)
            self._bounding_box(classifier_model, gray_img, scale_image, sbox.boundingboxs, batch_size_classifier, num_workers)

            all_bounding = all_bounding + sbox.boundingboxs

        return all_bounding
