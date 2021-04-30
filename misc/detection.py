import numpy as np
import cv2
import torch
import math
from sklearn.cluster import OPTICS
from .structure import BoundingBox, SentenceBox, CenterLine
from .nihongo import nihongo_class

class BoundingBoxDataset(object):
    def __init__(self, org_img, scale_wh, boundbox, output_size=(56,56)):
        self.org_img = org_img
        self.scale_wh = scale_wh
        self.boundbox = boundbox
        self.output_size = output_size
    def __getitem__(self, idx):
        bb = self.boundbox[idx]
        x1, y1, x2, y2 = bb.x1 * 2, bb.y1 * 2, bb.x2 * 2, bb.y2 * 2
        x1 = int(np.round(x1 * self.scale_wh[0]))
        y1 = int(np.round(y1 * self.scale_wh[1]))
        x2 = int(np.round(x2 * self.scale_wh[0]))
        y2 = int(np.round(y2 * self.scale_wh[1]))
        x1 = min(max(0, x1), self.org_img.shape[1])
        y1 = min(max(0, y1), self.org_img.shape[0])
        x2 = min(max(0, x2), self.org_img.shape[1])
        y2 = min(max(0, y2), self.org_img.shape[0])
        im = self.org_img[y1:y2,x1:x2]
        if im.shape[0]==0 or im.shape[1]==0:
            im = np.zeros((1,1))
        im = cv2.resize(im, self.output_size)
        im = im.reshape((1,self.output_size[1],self.output_size[0]))
        return im.astype(np.float32) / 255.
    def __len__(self):
        return len(self.boundbox)

class Detector(object):
    def __init__(self, use_cuda=True, word_threshold=0.01, class_threshold=0.25, low_gpu_memory=False):
        self.use_cuda = use_cuda
        self.word_threshold = word_threshold
        self.class_threshold = class_threshold
        self.low_gpu_memory = low_gpu_memory

    def _preprocess(self, hm_wd, hm_sent, hm_pos, simple_mode_lines=5, hm_dup=2.5, high_threshold=0.5, low_threshold=0.15):
        ln = np.clip((hm_sent*255),0,255).astype(np.uint8)
        lines = cv2.HoughLinesP(ln, rho=2, theta=np.pi/360, threshold=80, minLineLength=30, maxLineGap=15)
        center_ln = [] # Center line detection
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cl = CenterLine(hm_sent, hm_wd, hm_pos, [x1, y1], [x2, y2])
            center_ln.append(cl)
        lines = sorted(center_ln, key=lambda x: x.score)[::-1]
        drops = [] # NMS for line (drop duplicate line)
        for i in range(len(lines)):
            if i not in drops:
                cur = lines[i]
                for j in range(i+1, len(lines), 1):
                    otr = lines[j]
                    if cur.contain(otr) and j not in drops:
                        cur.score += otr.score
                        drops.append(j)
        lines = [l for i,l in enumerate(lines) if i not in drops]
        drops = [] # NMS for line (drop cross line)
        for i in range(len(lines)):
            if i not in drops:
                cur = lines[i]
                for j in range(i+1, len(lines), 1):
                    otr = lines[j]
                    if cur.cross(otr) and j not in drops:
                        drops.append(j)
        lines = [l for i,l in enumerate(lines) if i not in drops]
        if len(lines) <= simple_mode_lines: # simple image
            all_map = []
            for line in lines:
                out = np.zeros(hm_sent.shape, dtype=np.uint8)
                out = cv2.line(out, tuple(line.p1), tuple(line.p2), (255,255,255), line.maxw*2)
                all_map.append(out)
            return all_map, None
        else:
            wd_size_avg = (int(np.round(hm_pos[0][hm_pos[0]!=0].mean()*hm_pos[0].shape[1])),
                        int(np.round(hm_pos[1][hm_pos[1]!=0].mean()*hm_pos[1].shape[0])))
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, wd_size_avg)
            out = np.zeros(hm_sent.shape, dtype=np.uint8)
            for line in lines:
                out = cv2.line(out, tuple(line.p1), tuple(line.p2), (255,255,255), line.maxw*2)
            flt = hm_sent.copy() * hm_dup
            flt[hm_sent>high_threshold] = 1
            hm_sent_preprocessed = np.clip(flt+out/255,0,1)
            hm_sent_preprocessed[hm_sent<=low_threshold] = 0
            hm_sent_preprocessed = cv2.dilate(hm_sent_preprocessed, kernel)
            return None, hm_sent_preprocessed

    def _get_class(self, im, size=128):
        minmax = (im.min(), im.max())
        if minmax[1]-minmax[0] == 0:
            return np.array([])
        im = (im-minmax[0]) / (minmax[1]-minmax[0])
        sc = cv2.resize(im, (size,size), interpolation=cv2.INTER_NEAREST)
        clf = OPTICS(max_eps=5, metric='euclidean', min_cluster_size=75)
        a = []
        for x in range(sc.shape[0]):
            for y in range(sc.shape[1]):
                if sc[x][y] > 0.01:
                    a.append([x,y])
        b = clf.fit_predict(a)
        p = {v:k for k,v in enumerate(set(b))}
        b = [p[j] for j in b]
        c = np.zeros(sc.shape, dtype=np.int32)
        for i in range(len(b)):
            c[a[i][0],a[i][1]] = b[i]+1
        c = cv2.resize(c, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
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
        return np.array(sorted(maps, key=lambda x:-np.sum(x)))

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
        dp = detector_model
        if self.use_cuda:
            dp = torch.nn.DataParallel(detector_model)
            x = x.cuda()
            dp = dp.cuda()
        dp.eval()
        y = dp(x)

        hm_wd = y['hm_wd'].detach().cpu().numpy().reshape(256,256)
        hm_sent = y['hm_sent'].detach().cpu().numpy().reshape(256,256)
        hm_pos = y['of_size'].detach().cpu().numpy().reshape(2,256,256)
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
        if (not self.low_gpu_memory) or (not self.use_cuda):
            x = torch.tensor(x)
            dp = detector_model
            if self.use_cuda:
                dp = torch.nn.DataParallel(detector_model)
                x = x.cuda()
                dp = dp.cuda()
            dp.eval()
            y = dp(x)
            org_hm_wd = [y['hm_wd'][i].detach().cpu().numpy().reshape(256,256) for i in range(4)]
            org_hm_sent = [y['hm_sent'][i].detach().cpu().numpy().reshape(256,256) for i in range(4)]
            org_of_size = [y['of_size'][i].detach().cpu().numpy().reshape(2,256,256) / 2 for i in range(4)]
            del x, y
            if self.use_cuda:
                torch.cuda.empty_cache()
        else:
            org_hm_wd, org_hm_sent, org_of_size = [], [], []
            for i in range(4):
                _x = torch.tensor([x[i]])
                dp = detector_model
                if self.use_cuda:
                    dp = torch.nn.DataParallel(detector_model)
                    _x = _x.cuda()
                    dp = dp.cuda()
                dp.eval()
                y = dp(_x)
                org_hm_wd.append(y['hm_wd'][0].detach().cpu().numpy().reshape(256,256))
                org_hm_sent.append(y['hm_sent'][0].detach().cpu().numpy().reshape(256,256))
                org_of_size.append(y['of_size'][0].detach().cpu().numpy().reshape(2,256,256) / 2)
                del _x, y
                if self.use_cuda:
                    torch.cuda.empty_cache()
            del x

        hm_wd = np.zeros((512,512))
        hm_sent = np.zeros((512,512))
        hm_pos = np.zeros((2,512,512))
        hm_wd[0:256,0:256] = org_hm_wd[0]
        hm_wd[256:512,0:256] = org_hm_wd[1]
        hm_wd[0:256,256:512] = org_hm_wd[2]
        hm_wd[256:512,256:512] = org_hm_wd[3]
        hm_sent[0:256,0:256] = org_hm_sent[0]
        hm_sent[256:512,0:256] = org_hm_sent[1]
        hm_sent[0:256,256:512] = org_hm_sent[2]
        hm_sent[256:512,256:512] = org_hm_sent[3]
        hm_pos[:,0:256,0:256] = org_of_size[0]
        hm_pos[:,256:512,0:256] = org_of_size[1]
        hm_pos[:,0:256,256:512] = org_of_size[2]
        hm_pos[:,256:512,256:512] = org_of_size[3]
        return hm_wd, hm_sent, hm_pos

    def _detect16x(self, detector_model, gray_img_scaled):
        tmp = np.zeros((2048,2048))
        tmp[0:gray_img_scaled.shape[0],0:gray_img_scaled.shape[1]] = gray_img_scaled
        hm_wd = np.zeros((1024,1024))
        hm_sent = np.zeros((1024,1024))
        hm_pos = np.zeros((2,1024,1024))
        dp = detector_model
        if self.use_cuda:
            dp = torch.nn.DataParallel(detector_model)
            dp = dp.cuda()
        dp.eval()
        for ygrid_i in range(4):
            im = np.zeros((4,1,512,512))
            im[0,0] = tmp[512*ygrid_i:512*ygrid_i+512,0:512]
            im[1,0] = tmp[512*ygrid_i:512*ygrid_i+512,512:1024]
            im[2,0] = tmp[512*ygrid_i:512*ygrid_i+512,1024:1536]
            im[3,0] = tmp[512*ygrid_i:512*ygrid_i+512,1536:2048]
            x = np.clip(im / 255, 0.0, 1.0).astype(np.float32)
            if (not self.low_gpu_memory) or (not self.use_cuda):
                x = torch.tensor(x)
                if self.use_cuda:
                    x = x.cuda()
                y = dp(x)
                org_hm_wd = [y['hm_wd'][i].detach().cpu().numpy().reshape(256,256) for i in range(4)]
                org_hm_sent = [y['hm_sent'][i].detach().cpu().numpy().reshape(256,256) for i in range(4)]
                org_of_size = [y['of_size'][i].detach().cpu().numpy().reshape(2,256,256) / 4 for i in range(4)]
                del x, y
                if self.use_cuda:
                    torch.cuda.empty_cache()
            else:
                org_hm_wd, org_hm_sent, org_of_size = [], [], []
                for i in range(4):
                    _x = torch.tensor([x[i]])
                    dp = torch.nn.DataParallel(detector_model)
                    if self.use_cuda:
                        _x = _x.cuda()
                        dp = dp.cuda()
                    dp.eval()
                    y = dp(_x)
                    org_hm_wd.append(y['hm_wd'][0].detach().cpu().numpy().reshape(256,256))
                    org_hm_sent.append(y['hm_sent'][0].detach().cpu().numpy().reshape(256,256))
                    org_of_size.append(y['of_size'][0].detach().cpu().numpy().reshape(2,256,256) / 4)
                    del _x, y
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                del x

            hm_wd[256*ygrid_i:256*ygrid_i+256,0:256] = org_hm_wd[0]
            hm_wd[256*ygrid_i:256*ygrid_i+256,256:512] = org_hm_wd[1]
            hm_wd[256*ygrid_i:256*ygrid_i+256,512:768] = org_hm_wd[2]
            hm_wd[256*ygrid_i:256*ygrid_i+256,768:1024] = org_hm_wd[3]
            hm_sent[256*ygrid_i:256*ygrid_i+256,0:256] = org_hm_sent[0]
            hm_sent[256*ygrid_i:256*ygrid_i+256,256:512] = org_hm_sent[1]
            hm_sent[256*ygrid_i:256*ygrid_i+256,512:768] = org_hm_sent[2]
            hm_sent[256*ygrid_i:256*ygrid_i+256,768:1024] = org_hm_sent[3]
            hm_pos[:,256*ygrid_i:256*ygrid_i+256,0:256] = org_of_size[0]
            hm_pos[:,256*ygrid_i:256*ygrid_i+256,256:512] = org_of_size[1]
            hm_pos[:,256*ygrid_i:256*ygrid_i+256,512:768] = org_of_size[2]
            hm_pos[:,256*ygrid_i:256*ygrid_i+256,768:1024] = org_of_size[3]

        return hm_wd, hm_sent, hm_pos

    def _get_maps(self, detector_model, gray_img, dpi, min_word_size_cm):
        if dpi == 0:
            img_size = max(gray_img.shape)
            if img_size <= 512:
                detect_size = 512
            elif img_size <= 1024:
                detect_size = 1024
            else:
                detect_size = 2048
            div_size = min(img_size, 2048)
        else:
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

        with torch.no_grad():
            if detect_size == 512:
                hm_wd, hm_sent, hm_pos = self._detect1x(detector_model, gray_img_scaled)
            elif detect_size == 1024:
                hm_wd, hm_sent, hm_pos = self._detect4x(detector_model, gray_img_scaled)
            elif detect_size == 2048:
                hm_wd, hm_sent, hm_pos = self._detect16x(detector_model, gray_img_scaled)

        hm_wd[np.mean(hm_pos, axis=0) < 0.01] = 0
        hm_sent[np.mean(hm_pos, axis=0) < 0.01] = 0
        return pix_image, scale_image, hm_wd, hm_sent, hm_pos

    def _find_best_dpi(self, detector_model, gray_img, dpi, min_word_size_cm):
        tests = []
        for testdpi in (72,100,150,200,300):
            res = self._get_maps(detector_model, gray_img, testdpi, min_word_size_cm)
            tests.append((testdpi, np.sum(res[3] > 0.01) / (res[3].shape[0]*res[3].shape[1]), res))
        result = sorted(tests, key=lambda x:x[1])[-1]
        return result[0], result[2]

    def detect_image(self, detector_model, gray_img, dpi=72, min_word_size_cm=0.5):
        min_bound = int(np.round(min_word_size_cm * dpi / 2.54))

        if dpi >= 0:
            pix_image, scale_image, hm_wd, hm_sent, hm_pos = self._get_maps(detector_model, gray_img, dpi, min_word_size_cm)
        else:
            dpi, (pix_image, scale_image, hm_wd, hm_sent, hm_pos) = self._find_best_dpi(detector_model, gray_img, dpi, min_word_size_cm)

        all_map, hm_sent_preprocessed = self._preprocess(hm_wd, hm_sent, hm_pos)
        if hm_sent_preprocessed is not None:
            class_map = self._get_class(hm_sent_preprocessed)
            all_map = self._get_map(class_map)
            all_map = self._filt_map(all_map)

        sent_box = []

        for i, now_map in enumerate(all_map):
            clz_wd = hm_wd.copy()
            clz_wd[now_map == 0] = 0
            clz_wd = (clz_wd - np.min(clz_wd)) / (np.max(clz_wd) - np.min(clz_wd))
            sbox = SentenceBox(self.word_threshold)
            sbox.make_boundingbox(clz_wd, hm_pos, min_bound)
            if len(sbox.boundingboxs) > 0:
                sent_box.append(sbox)

        return dpi, sent_box, gray_img, scale_image, hm_wd

    def _bounding_box(self, classifier_model, gray_img, scale_image, boundings, batch_size_classifier, num_workers):
        dataset = BoundingBoxDataset(gray_img, scale_image, boundings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_classifier, shuffle=False, num_workers=num_workers)
        with torch.no_grad():
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
                    boundings[num_pred].set_prediction(v.copy())
                    num_pred += 1
                del x
            del dp
            if self.use_cuda:
                torch.cuda.empty_cache()

    def bounding_box(self, classifier_model, detection, batch_size_classifier=64, num_workers=2, repeat_box=1):
        if self.low_gpu_memory:
            batch_size_classifier = batch_size_classifier//8

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

        all_bounding = [b for b in all_bounding if b.classifiercore() > self.class_threshold]
        return all_bounding
