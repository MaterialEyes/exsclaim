import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import itertools

class ConnectComponents():
    def __init__(self,bboxes,classes,confidences,width,height):
        flag = False
        master_image_bboxes,subfigure_label_bboxes = self.RearrangeBBoxes(bboxes,classes,confidences)
        print("\n⦿ Master Image: {}, Subfigure Label: {}".format(len(master_image_bboxes), len(subfigure_label_bboxes)))
        if len(master_image_bboxes) == len(subfigure_label_bboxes):
            match_result = self.FindOptimalMatch(master_image_bboxes,subfigure_label_bboxes,width,height)
        elif len(master_image_bboxes) > len(subfigure_label_bboxes):
            flag = "master"
            indices = list(range(len(master_image_bboxes)))
            perm = itertools.combinations(indices,len(subfigure_label_bboxes))
            match_results = []
            best_scores = []
            for index in list(perm):
                selected_master_image_bboxes = []
                for i in range(len(index)):
                    selected_master_image_bboxes.append(master_image_bboxes[index[i]])
                match_result = self.FindOptimalMatch(selected_master_image_bboxes,subfigure_label_bboxes,width,height)
                if match_result:
                    best_scores.append(match_result[2][-1])
                    unselected_indices = []
                    for idx in indices:
                        if idx not in index:
                            unselected_indices.append(idx)
                    match_results.append([unselected_indices,match_result])
            if match_results:
                match_result = match_results[np.argmax(best_scores)]
        else:
            flag = "subfigure"
            indices = list(range(len(subfigure_label_bboxes)))
            perm = itertools.combinations(indices,len(master_image_bboxes))
            match_results = []
            best_scores = []
            for index in list(perm):
                selected_subfigure_label_bboxes = []
                for i in range(len(index)):
                    selected_subfigure_label_bboxes.append(subfigure_label_bboxes[index[i]])
                match_result = self.FindOptimalMatch(master_image_bboxes,selected_subfigure_label_bboxes,width,height)
                if match_result:
                    best_scores.append(match_result[2][-1])
                    unselected_indices = []
                    for idx in indices:
                        if idx not in index:
                            unselected_indices.append(idx)
                    match_results.append([unselected_indices,match_result])
            if match_results:
                match_result = match_results[np.argmax(best_scores)]
            
        
#         print("best_scores is {}".format(best_scores))
        if match_result:
            if len(match_result) == 2:
                remaining_indices = match_result[0]
#                 print("remaining_indices is {}".format(remaining_indices))
                match_result = match_result[1]
                
            if len(match_result) == 4:
                cate_bboxes_master, cate_bboxes_subfigure, scores, indexes = match_result
                assert len(cate_bboxes_master) == len(cate_bboxes_subfigure)
                index = indexes[scores[-1]]
                matched_bboxes = []
                for i in range(len(cate_bboxes_subfigure)):
                    item = [cate_bboxes_subfigure[i][0],cate_bboxes_subfigure[i][1],cate_bboxes_master[index[i]][0],cate_bboxes_master[index[i]][1], cate_bboxes_subfigure[i][2], cate_bboxes_master[index[i]][2]]
                    matched_bboxes.append(item)
                unpaired_bboxes = []
                if flag == "master":
                    for index in remaining_indices:
                        item = [None,None,master_image_bboxes[index][0],master_image_bboxes[index][1], None, master_image_bboxes[index][2]]
                        unpaired_bboxes.append(item)
                elif flag == "subfigure":
                    for index in remaining_indices:
                        item = [subfigure_label_bboxes[index][0],subfigure_label_bboxes[index][1],None,None,subfigure_label_bboxes[index][2], None]
                        unpaired_bboxes.append(item)
                matched_bboxes.extend(unpaired_bboxes)
                self.matched_bboxes = matched_bboxes
                    
            else:
                raise ValueError("match_result should contain 4 components, but found {}".format(len(match_result)))
                
        else:
            raise NotImplementedError("This image is too difficult for our current model")
    
        
            
    
    def FindOptimalMatch(self, master_image_bboxes,subfigure_label_bboxes,width,height,top_k = 10):
        posi_bboxes_master,cate_bboxes_master = self.GetSortedBBoxes(master_image_bboxes)
        posi_bboxes_subfigure,cate_bboxes_subfigure = self.GetSortedBBoxes(subfigure_label_bboxes)
        posi_bboxes_master = self.CorrectBBoxes(posi_bboxes_master,width,height)
        posi_bboxes_subfigure = self.CorrectBBoxes(posi_bboxes_subfigure,width,height)
        scores,indexes = self.ScoreCombinations(bboxes_1=posi_bboxes_subfigure, bboxes_2=posi_bboxes_master, top_k=top_k)
        if indexes:
            return cate_bboxes_master, cate_bboxes_subfigure, scores, indexes
        else:
            return False
            
    
    def RearrangeBBoxes(self,bboxes,classes,confidences):
        master_image_bboxes = []
        subfigure_label_bboxes = []
        for i in range(len(bboxes)):
            tmp = []
            tmp.append(classes[i])
            tmp.append(bboxes[i])
            tmp.append(confidences[i])
            if classes[i] < 8:
                master_image_bboxes.append(tmp)
            else:
                subfigure_label_bboxes.append(tmp)
        return master_image_bboxes,subfigure_label_bboxes
    
    def GetSortedBBoxes(self, bboxes):
    
        # sort bboxes
        for i in range(len(bboxes)):
            for j in range(i+1,len(bboxes)):
                if bboxes[j][1][0] < bboxes[i][1][0]:
                    tmp = bboxes[j]
                    bboxes[j] = bboxes[i]
                    bboxes[i] = tmp
                elif bboxes[j][1][0] == bboxes[i][1][0] and bboxes[j][1][1] < bboxes[i][1][1]:
                    tmp = bboxes[j]
                    bboxes[j] = bboxes[i]
                    bboxes[i] = tmp

        # decouple bboxes and labels
        posi_bboxes = []
        cate_bboxes = []
        for i in range(len(bboxes)):
#             cate_bboxes.append(bboxes[i][0])
            posi_bboxes.append(bboxes[i][1])
        return posi_bboxes,bboxes
    
    def CorrectBBoxes(self, bboxes,width,height):
        new_bboxes=[]
        for i in range(len(bboxes)):
            x1,y1,x2,y2 = bboxes[i]
            x1 = (max(x1,0))/width
            y1 = (max(y1,0))/height
            x2 = (min(x2,width))/width
            y2 = (min(y2,height))/height
            new_bboxes.append([x1,y1,x2,y2])
        return new_bboxes
    
    # bboxes_1="Subfigure Label", bboxes_2="Master Image"
    def ScoreCombinations(self, bboxes_1, bboxes_2, top_k = 10):
        def Score(bbox1,bbox2,bbox12=True, overlap_thresh=0.1):
            x1 = max([bbox1[0],bbox2[0]])
            y1 = max([bbox1[1],bbox2[1]])
            x2 = min([bbox1[2],bbox2[2]])
            y2 = min([bbox1[3],bbox2[3]])

            overlap_area = max(x2-x1,0)*max(y2-y1,0)
            bbox_area = max(bbox1[2]-bbox1[0],0)*max(bbox1[3]-bbox1[1],0)
            assert bbox_area>0, "the area of the bbox should be above 0, but found {}".format(bbox1)
            if overlap_area/bbox_area > overlap_thresh:
                return 1.0
            else:
                def f(dist):
                    return (dist-2)**2/4
                delta_x = min([abs(bbox1[0]-bbox2[0]),abs(bbox1[0]-bbox2[2]),abs(bbox1[2]-bbox2[0]),abs(bbox1[2]-bbox2[2])])
                delta_y = min([abs(bbox1[1]-bbox2[1]),abs(bbox1[1]-bbox2[3]),abs(bbox1[3]-bbox2[1]),abs(bbox1[3]-bbox2[3])])
                dist = delta_x+delta_y
                return f(dist)

        def CalTotalScore(bboxes1,bboxes2,index):
            total_score = 1.0
            for i in range(len(index)):
                if total_score<0.4:
                    return 0.0
                total_score= total_score*Score(bboxes1[i],bboxes2[index[i]])
            return total_score

        bboxes1 = bboxes_1
        bboxes2 = bboxes_2

        indices = list(range(len(bboxes1)))
        perm = itertools.permutations(indices)
        scores = np.zeros(top_k)
        indexes = {}
        for index in list(perm):
            score = CalTotalScore(bboxes1,bboxes2,index)
            if score > scores[0]:
                if scores[0] in indexes:
                    del indexes[scores[0]]
                scores[0] = score
                scores = sorted(scores)
                indexes[score] = index
        return scores,indexes
