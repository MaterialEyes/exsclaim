import numpy as np
from PIL import Image


def FindingMissingMasterImages(matched_bboxes,width,height):
    grain_size=20
    min_area = 1.0*width*height/(len(matched_bboxes)**2)
#     print(FillingFinished(matched_bboxes))
    while not FillingFinished(matched_bboxes):
        for i in range(len(matched_bboxes)):
            if matched_bboxes[i][2]:
                pass
            else:
                break
        new_image,target_bbox = DrawingBinaryImage(width,height,matched_bboxes,i)
        matched_bboxes[i] = FillingMissingMasterImage(new_image,target_bbox, min_area,grain_size)
    return matched_bboxes
        
        
        
def DrawingBinaryImage(width,height,matched_bboxes,index):
    new_image = Image.new(mode="RGB",size=(width,height), color=(255,255,255))
    for i in range(len(matched_bboxes)):
        if i == index:
            continue
        else:
            _, subfigure_label_bbox, _,master_image_bbox = matched_bboxes[i]
            if subfigure_label_bbox:
                x1,y1,x2,y2 = subfigure_label_bbox
                black_box = Image.new(mode="RGB", size=[int(x2-x1),int(y2-y1)],color=0)
                new_image.paste(black_box,box=(int(x1),int(y1)))
            else:
                raise NotImplementedError("No missing subfigure label is allowed")
            if master_image_bbox:
                x1,y1,x2,y2 = master_image_bbox
                black_box = Image.new(mode="RGB", size=[int(x2-x1),int(y2-y1)],color=0)
                new_image.paste(black_box,box=(int(x1),int(y1)))
            else:
                pass
    return new_image,matched_bboxes[index]


def FillingMissingMasterImage(new_image,target_bbox, min_area,grain_size=20):
    def DistMasterSubfigure(master_bbox,subfigure_bbox,width,height):
        x1 = max(master_bbox[0],subfigure_bbox[0])
        y1 = max(master_bbox[1],subfigure_bbox[1])
        x2 = min(master_bbox[2],subfigure_bbox[2])
        y2 = min(master_bbox[3],subfigure_bbox[3])
        overlap_area = max(0,x2-x1)*max(0,y2-y1)
        subfigure_area = (subfigure_bbox[2]-subfigure_bbox[0])*(subfigure_bbox[3]-subfigure_bbox[1])
        if overlap_area/subfigure_area >0.5:
            return 1.0
        else:
            def f(dist):
                return (dist-2)**2/4
            bbox1 = master_bbox
            bbox2 = subfigure_bbox
            delta_x = min([abs(bbox1[0]-bbox2[0]),abs(bbox1[0]-bbox2[2]),abs(bbox1[2]-bbox2[0]),abs(bbox1[2]-bbox2[2])])
            delta_y = min([abs(bbox1[1]-bbox2[1]),abs(bbox1[1]-bbox2[3]),abs(bbox1[3]-bbox2[1]),abs(bbox1[3]-bbox2[3])])
            dist = delta_x/width+delta_y/height
            return f(dist)
    
    source_image = np.array(new_image.convert("L"))
    row,col = source_image.shape
    grid = np.linspace(0,1,grain_size)
    bboxes = []
    areas = []
    for start_row_index in range(len(grid)):
        for start_col_index in range(len(grid)):
            for end_row_index in range(start_row_index+1, len(grid)):
                for end_col_index in range(start_col_index+1, len(grid)):
                    row_s = int(grid[start_row_index]*row)
                    col_s = int(grid[start_col_index]*col)
                    row_e = int(grid[end_row_index]*row)
                    col_e = int(grid[end_col_index]*col)
                    region = source_image[row_s:row_e,col_s:col_e]
                    if np.amin(region) > 0 and (col_e-col_s)*(row_e-row_s)>min_area:
                        bboxes.append([col_s,row_s,col_e,row_e])
                        areas.append((col_e-col_s)*(row_e-row_s))
    subfigure_label_cls, subfigure_label_bbox,master_image_cls,master_image_bbox = target_bbox
    close_bboxes=[]
    close_areas=[]
    for i in range(len(bboxes)):
        score = DistMasterSubfigure(bboxes[i],subfigure_label_bbox,col,row)
        if score > 0.9:
            close_bboxes.append(bboxes[i])
            close_areas.append(areas[i])
    
    if close_areas:
        master_image_bbox = close_bboxes[np.argmax(close_areas)]
    master_image_cls = 7 #("unclear")
    return subfigure_label_cls, subfigure_label_bbox,master_image_cls,master_image_bbox

def FillingFinished(matched_bboxes):
    for matched_bbox in matched_bboxes:
        subfigure_label_cls, subfigure_label_bbox,master_image_cls,master_image_bbox = matched_bbox
        if not master_image_cls:
            return False
    return True

def AllSubfigureLabelsFound(matched_bboxes):
    for matched_bbox in matched_bboxes:
        subfigure_label_cls, subfigure_label_bbox,master_image_cls,master_image_bbox = matched_bbox
        if not subfigure_label_cls:
            return False
    return True