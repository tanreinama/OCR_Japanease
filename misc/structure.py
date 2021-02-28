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
        idx = np.argmax(self.prediction[1:len(nihongo_class)]) + 1
        return idx, self.prediction[idx]
    def classifiercore(self):
        return self.word()[1]
    def score(self):
        return self.detectionscore * self.classifiercore()


class SentenceBox(object):
    def __init__(self, word_threshold):
        self.boundingboxs = []
        self.word_threshold = word_threshold

    def _conv3_filter(self, img, pos):
        points = []
        rects = []
        sizes = []
        for y in range(1,img.shape[0]-1):
            for x in range(1,img.shape[1]-1):
                if img[y,x]>self.word_threshold and img[y,x]>img[y-1,x] and img[y,x]>img[y-1,x-1] and img[y,x]>img[y-1,x+1] and img[y,x]>img[y,x-1] and img[y,x]>img[y,x+1] and img[y,x]>img[y+1,x] and img[y,x]>img[y+1,x-1] and img[y,x]>img[y+1,x+1]:
                    points.append((x,y))
                    w, h = pos[0][y,x], pos[1][y,x]
                    sizes.append(max(w,h))
                    w, h = 1+w*img.shape[1], 1+h*img.shape[0]
                    offw, offh = int(np.round(w)), int(np.round(h))
                    rects.append((x-offw,y-offh,x+offw,y+offh))
        return points, rects, sizes

    def make_boundingbox(self, hm_wd, hm_pos, min_bound=12, resize_val=1.1, aspect_val=1.25, dup_threathold=0.033):
        pos, rcts, sizes = self._conv3_filter(hm_wd, hm_pos)
        min_bound = min_bound // 2
        for p, r, s in zip(pos, rcts, sizes):
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
            if s > dup_threathold:
                w2 = int(np.round((x2-x1) * aspect_val))
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

class CenterLine:
    def __init__(self, hm_sent, hm_word, hm_pos, p1, p2):
        assert hm_sent.shape == hm_word.shape and hm_sent.shape[0] == hm_pos.shape[1] and hm_sent.shape[1] == hm_pos.shape[2], 'Invalid heatmap'
        assert p1 != p2, 'Invalid point'
        self.hm_sent = hm_sent
        self.hm_word = hm_word
        self.hm_pos = hm_pos
        self.p1 = p1
        self.p2 = p2
        self.score = 0
        w = []
        dx = 1 if self.p2[0]>=self.p1[0] else -1
        dy = 1 if self.p2[1]>=self.p1[1] else -1
        for x in range(self.p1[0], self.p2[0]+dx, dx):
            for y in range(self.p1[1], self.p2[1]+dy, dy):
                m = max(self.hm_pos[0][y][x]*self.hm_pos.shape[0], self.hm_pos[1][y][x]*self.hm_pos.shape[1])
                w.append(int(np.round(m)))
        self.maxw = np.max(w)
        self.stdw = np.std(w)
        self.score = self._score()
    def _intersect(self, p1, p2, p3, p4):
        tc1 = float(p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        tc2 = float(p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        td1 = float(p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        td2 = float(p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
        return tc1*tc2<0 and td1*td2<0
    def _distance(self, p1, p2, other):
        ab = [p2[0] - p1[0], p2[1] - p1[1]]
        ap = [other[0] - p1[0], other[1] - p1[1]]
        bp = [other[0] - p2[0], other[1] - p2[1]]
        if (ab[0]==0 and ab[1] == 0) or (bp[0]==0 and bp[1] == 0) or (ap[0]==0 and ap[1] == 0):
            return 0
        d = np.cross(ab, ap)
        l = np.sqrt(ab[0]**2+ab[1]**2)
        e = d / l # 点と線の距離
        v = [ab[0]/l, ab[1]/l]
        m = np.dot(v, ap)
        if m >= 0 and m <= l:
            return abs(e)
        return min(np.sqrt(ap[0]**2+ap[1]**2), np.sqrt(bp[0]**2+bp[1]**2))
    def _score(self):
        filt = np.zeros(self.hm_sent.shape, dtype=np.uint8)
        filt = cv2.line(filt, tuple(self.p1), tuple(self.p2), (255,255,255), self.maxw)
        return np.sum((self.hm_word > 0.015)[filt!=0])
    def dist(self, other):
        if self._intersect(self.p1, self.p2, other.p1, other.p2):
            return 0
        d1 = min(self._distance(self.p1, self.p2, other.p1), self._distance(self.p1, self.p2, other.p2))
        d2 = min(self._distance(other.p1, other.p2, self.p1), self._distance(other.p1, other.p2, self.p2))
        return min(d1, d2)
    def cross(self, other):
        return self.dist(other) < (self.maxw + other.maxw)/2
    def contain(self, other):
        if self._intersect(self.p1, self.p2, other.p1, other.p2):
            return True
        d = max(self._distance(self.p1, self.p2, other.p1), self._distance(self.p1, self.p2, other.p2))
        return d < self.maxw
