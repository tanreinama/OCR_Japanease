import numpy as np

def non_max_suppression(boxes, overlapThresh=0.2):
    if len(boxes) == 0:
        return []

    sorted_box = sorted(boxes, key=lambda x:x.score())[::-1]
    ignore_flg = [False] * len(sorted_box)

    for i in range(len(sorted_box)):
        if not ignore_flg[i]:
            for j in range(i+1,len(sorted_box),1):
                r1 = sorted_box[i]
                r2 = sorted_box[j]
                if r1.x1 <= r2.x2 and r2.x1 <= r1.x2 and r1.y1<= r2.y2 and r2.y1 <= r1.y2:
                    w = max(0, min(r1.x2,r2.x2) - max(r1.x1,r2.x1))
                    h = max(0, min(r1.y2,r2.y2) - max(r1.y1,r2.y1))
                    if w * h > (r2.x2-r2.x1)*(r2.y2-r2.y1)*overlapThresh:
                        ignore_flg[j] = True

    return [sorted_box[i] for i in range(len(sorted_box)) if not ignore_flg[i]]

def column_wordlines(bbox, overlapThresh=0.1, overlapThresh_line=0.6):
    def _1dim_non_suppression(ranges, overlapThresh):
        if len(ranges) == 0:
            return []

        ignore_flg = [False] * len(ranges)

        for i in range(len(ranges)):
            if not ignore_flg[i]:
                for j in range(i+1,len(ranges),1):
                    r1 = ranges[i]
                    r2 = ranges[j]
                    w = max(0, min(r1[1],r2[1]) - max(r1[0],r2[0]))
                    if w > (r2[1]-r2[0])*overlapThresh:
                        ignore_flg[j] = True

        return [ranges[i] for i in range(len(ranges)) if not ignore_flg[i]]

    box_range_x = [(b.x1,b.x2) for b in bbox]
    box_range_y = [(b.y1,b.y2) for b in bbox]
    cols = _1dim_non_suppression(box_range_x, overlapThresh)
    rows = _1dim_non_suppression(box_range_y, overlapThresh)
    stocked_flg = [False] * len(bbox)
    lines = []
    if len(cols) < len(rows): # 縦書き
        for c in cols:
            stocks = []
            for i in range(len(bbox)):
                if not stocked_flg[i]:
                    if c[0] < bbox[i].x2 and c[1] > bbox[i].x1:
                        w = max(0, min(c[1],bbox[i].x2) - max(c[0],bbox[i].x1))
                        if w > (bbox[i].x2-bbox[i].x1)*overlapThresh_line:
                            stocks.append(bbox[i])
                            stocked_flg[i] = True
            lines.append(sorted(stocks, key=lambda x:x.y1))
        lines = sorted(lines, key=lambda x: np.mean([y.x1 for y in x]))
    else: # 横書き
        for r in rows:
            stocks = []
            for i in range(len(bbox)):
                if not stocked_flg[i]:
                    if r[0] < bbox[i].y2 and r[1] > bbox[i].y1:
                        h = max(0, min(r[1],bbox[i].y2) - max(r[0],bbox[i].y1))
                        if h >= (bbox[i].y2-bbox[i].y1)*overlapThresh_line:
                            stocks.append(bbox[i])
                            stocked_flg[i] = True
            lines.append(sorted(stocks, key=lambda x:x.x1))
        lines = sorted(lines, key=lambda x: np.mean([y.y1 for y in x]))
    return lines
