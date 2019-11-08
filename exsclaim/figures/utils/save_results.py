from PIL import Image, ImageDraw, ImageFont
from .match_bboxes import ConnectComponents
import os
from .fix_missing_pair import FindingMissingMasterImages, AllSubfigureLabelsFound, FillingFinished
class SavingResults():
    def __init__(self, bboxes, classes,confidences, image, fs_class_names, \
                 result_dir, caption_prior=False, \
                 figure_separator_evaluator=None,json_dict={},\
                 json_dict_key = "figure_separator_results"):
        self.bboxes = bboxes
        self.classes = classes
        self.confidences = confidences
        self.json_dict = json_dict
        self.json_dict_key = json_dict_key
        self.image = image
        self.fs_class_names = fs_class_names
        self.result_dir = result_dir
        self.caption_prior = caption_prior
        self.figure_separator_evaluator = figure_separator_evaluator
        self.test_img = Image.open(image).convert("RGB")
        self.width,self.height = self.test_img.size
        self.MainFunc()
        
    def Visualization(self,matched_bboxes):
        width,height=self.width,self.height
        result_image = Image.new(mode="RGB",size=(300,len(matched_bboxes)*100+50))
        draw = ImageDraw.Draw(result_image)
        font = ImageFont.load_default()
        # draw the headline
        text = "description"
        draw.text((10,10),text,fill="white",font=font)
        text = "subfigure label"
        draw.text((110,10),text,fill="white",font=font)
        if self.caption_prior:
            draw.text((110,30),"({})".format(self.figure_separator_evaluator.GetPriorFromCaption(self.image.split("/")[-1])),fill="white",font=font)
        text = "master image"
        draw.text((210,10),text,fill="white",font=font)
        
        for i in range(len(matched_bboxes)):
            subfigure_label_cls, subfigure_label_bbox,master_image_cls,master_image_bbox,subfigure_label_conf,master_image_conf = matched_bboxes[i]
            # print description
            if subfigure_label_cls and subfigure_label_bbox:
                draw.text((10,50+100*i+10),self.fs_class_names[subfigure_label_cls], fill="white",font=font)
                draw.text((10,50+100*i+30),"({})".format(subfigure_label_conf), fill="white",font=font)
#                 text = self.fs_class_names[subfigure_label_cls] + "({}):".format(subfigure_label_conf)
            else:
                draw.text((10,50+100*i+10),"Unknown:", fill="white",font=font)
#                 text = "Unknown:"
            if master_image_cls and master_image_bbox:
                draw.text((10,50+100*i+50),self.fs_class_names[master_image_cls], fill="white",font=font)
                draw.text((10,50+100*i+70),"({})".format(master_image_conf), fill="white",font=font)
#                 text += self.fs_class_names[master_image_cls]+"({})".format(master_image_conf)
            else:
                draw.text((10,50+100*i+50),"Unknown", fill="white",font=font)
#                 text += "Unknown"
#             draw.text((10,50+100*i+10),text,fill="white",font=font)
            # paste subfigure label
            if subfigure_label_cls and subfigure_label_bbox:
                x1,y1,x2,y2 = subfigure_label_bbox
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,width-1)
                y2 = min(y2,height-1)
                sub_figure = self.test_img.crop((x1,y1,x2,y2))
                sub_figure = sub_figure.resize((80,80))
                result_image.paste(sub_figure,box=(110,50+100*i+10))
                        
            # paste master image
            if master_image_cls and master_image_bbox:
                x1,y1,x2,y2 = master_image_bbox
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,width-1)
                y2 = min(y2,height-1)
                sub_figure = self.test_img.crop((x1,y1,x2,y2))
                sub_figure = sub_figure.resize((80,80))
                result_image.paste(sub_figure,box=(210,50+100*i+10))
        del draw
        inpt_img_name = self.image.split("/")[-1].split(".")[0]+"_input.png"
        self.test_img.save(os.path.join(self.result_dir,inpt_img_name))
        result_img_name = self.image.split("/")[-1].split(".")[0]+"_output.png"
        result_image.save(os.path.join(self.result_dir,result_img_name))
                
        
    def Documentation(self, matched_bboxes):
        width,height=self.width,self.height
        image_info = {}
        image_info["figure_name"] = self.image.split("/")[-1]
        image_info["master_images"] = []
        image_info["unassigned"] = []
        
        for i in range(len(matched_bboxes)):
            subfigure_label_cls, subfigure_label_bbox, master_image_cls, master_image_bbox, subfigure_label_conf, master_image_conf = matched_bboxes[i]
            if master_image_cls and master_image_bbox:
                image_info_master_image = {}
                image_info_master_image["classification"] = self.fs_class_names[master_image_cls]
                image_info_master_image["score"] = master_image_conf
                image_info_master_image["geometry"] = []
                if subfigure_label_cls and subfigure_label_bbox:
                    image_info_master_image["subfigure_label"] = {}
                    image_info_master_image["subfigure_label"]["text"] = self.fs_class_names[subfigure_label_cls]
                    image_info_master_image["subfigure_label"]["score"] = subfigure_label_conf
                    image_info_master_image["subfigure_label"]["geometry"] = []
            else:
                image_info_subfigure_label = {}
                image_info_subfigure_label["text"] = self.fs_class_names[subfigure_label_cls]
                image_info_subfigure_label["score"] = subfigure_label_conf
                image_info_subfigure_label["geometry"] = []
                
                
            # record subfigure label
            if subfigure_label_cls and subfigure_label_bbox:
                x1,y1,x2,y2 = subfigure_label_bbox
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,width-1)
                y2 = min(y2,height-1)
                for x in [x1,x2]:
                    for y in [y1,y2]:
                        subfigure_geometry = {}
                        subfigure_geometry["x"] = x
                        subfigure_geometry["y"] = y
                        if master_image_cls and master_image_bbox:
                            image_info_master_image["subfigure_label"]["geometry"].append(subfigure_geometry)
                        else:
                            image_info_subfigure_label["geometry"].append(subfigure_geometry)
                        
            # record master image
            if master_image_cls and master_image_bbox:
                x1,y1,x2,y2 = master_image_bbox
                x1 = max(x1,0)
                y1 = max(y1,0)
                x2 = min(x2,width-1)
                y2 = min(y2,height-1)
                for x in [x1,x2]:
                    for y in [y1,y2]:
                        image_info_master_image_geometry = {}
                        image_info_master_image_geometry["x"] = x
                        image_info_master_image_geometry["y"] = y
                        image_info_master_image["geometry"].append( image_info_master_image_geometry)
            if master_image_cls and master_image_bbox:
                if subfigure_label_cls and subfigure_label_bbox:
                    image_info["master_images"].append(image_info_master_image)
                else: 
                    image_info["unassigned"].append(image_info_master_image)
            else:
                image_info["unassigned"].append(image_info_subfigure_label)
        self.json_dict[self.json_dict_key].append(image_info)
    
    def MainFunc(self):
        try:
            matched_bboxes = ConnectComponents(self.bboxes, \
                                               self.classes, \
                                               self.confidences, \
                                               float(self.width), \
                                               float(self.height)).matched_bboxes
#             if len(matched_bboxes) == self.figure_separator_evaluator.GetPriorFromCaption(self.image.split("/")[-1]) \
#             and AllSubfigureLabelsFound(matched_bboxes) \
#             and not FillingFinished(matched_bboxes):
#                 matched_bboxes = FindingMissingMasterImages(matched_bboxes,self.width,self.height)
            self.Visualization(matched_bboxes)
            self.Documentation(matched_bboxes)
        except NotImplementedError:
            print("failed to match bboxes for image {}".format(self.image.split("/")[-1]))