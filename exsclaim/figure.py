import os
import yaml
import json
import glob
import matplotlib.pyplot as plt
from .figures.utils.utils import *
from .figures.utils.save_results import SavingResults
from .figures.models.yolov3 import *
from torch.autograd import Variable


def load_model(model_path=str) -> "figure_separator_model":
    """
    Opens and extracts model snapshot from configuration file + checkpoints

    Args:
        model_path: A path to the model files

    Returns:
        figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
    """
    # Fixed model paths/parameters 
    gpu = 0
    detection_threshold = None
    # checkpoint  = model_path + "checkpoints/snapshot930.ckpt"
    checkpoint  = model_path + "checkpoints/snapshot20000.ckpt"
    config_file = model_path + "config/yolov3_eval.cfg"

    # Begin model load procedure
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    image_size = cfg['TEST']['IMGSIZE']
    model = YOLOv3(cfg['MODEL'])

    confidence_threshold = cfg['TEST']['CONFTHRE']
    nms_threshold = cfg['TEST']['NMSTHRE']

    if detection_threshold:
        confidence_threshold = detection_threshold

    if gpu > 0:
        model.cuda(args[gpu])
        # print("loading checkpoint %s" % (checkpoint))
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    else:
        # print("loading checkpoint %s" % (checkpoint))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])

    return (model, confidence_threshold, nms_threshold, image_size, gpu)


def get_figure_paths(search_query: dict) -> list:
    """
    Get a list of paths to figures extracted using the search_query

    Args:
        search_query: A query json
    Returns:
        A list of figure paths
    """
    extensions = ['.png','jpg','.gif']
    paths = []
    for ext in extensions:
        paths+=glob.glob(search_query['results_dir']+'figures/*'+ext)
    return paths


def write_object_dictionary(object_data) -> "figure_dict":
    """
    Find individual image objects within a figure and classify based on functionality

    Args:
        object_data: (outputs, info_image) from evaluation of PyTorch model

    Returns:
        figure_dict: A dictionary with classified image_objects extracted from figure
    """
    """ converts image_data to MaterialEyes JSON """
    outputs, info_image = object_data
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
     
    classes = {0: "master_images", 1: "master_images", 2: "subfigure_labels", 3: "scale_bar_labels"}
    #for name, ID, color in zip(coco_class_names, coco_class_ids, coco_class_colors):
    #    classes[ID] = (name, color)
    bboxes = {"master_images": [], "subfigure_labels": [], "scale_bar_labels": []}
    # Only proceed if objects have been detected!
    if type(outputs[0]) == torch.Tensor: 
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            y_1, x_1, y_2, x_2 = yolobox2label([y1, x1, y2, x2], info_image)
            y_1 = max(int(y_1), 0)
            x_1 = max(int(x_1), 0)
            y_2 = min(int(y_2), info_image[0])
            x_2 = min(int(x_2), info_image[1])
            location = [{"x" : x_1, "y" : y_1}, {"x" : x_1, "y" : y_2},
                        {"x" : x_2, "y" : y_2}, {"x" : x_2, "y" : y_1}] 
            # object_entry = {"geometry" : location, "confidence" : float(cls_conf), "text": ""}
            object_entry_image = {"geometry" : location, "classification": None}
            object_entry_label = {"geometry" : location, "text": ""}
            box_class = classes[int(cls_pred)]

            #  Create 'classification' key for image objects and 'text' key for label objects 
            if box_class.split("_")[-1] == 'images':
                bbox = bboxes.get(box_class, [])
                bbox.append(object_entry_image)
            if box_class.split("_")[-1] == 'labels':
                bbox = bboxes.get(box_class, [])
                bbox.append(object_entry_label)
            bboxes[box_class] = bbox 
    return bboxes


def extract_image_objects(figure_separator_model=tuple, figure_path=str) -> "figure_dict":
    """
    Find individual image objects within a figure and classify based on functionality

    Args:
        figure_separator_model: A tuple (model, confidence_threshold, nms_threshold, img_size, gpu)
        figure_path: A path to the figure to separate

    Returns:
        figure_dict: A dictionary with classified image_objects extracted from figure
    """
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()
    json_dict = {}
    json_dict["figure_separator_results"] = []

    separations_path = figure_path.split("figures")[0]+"separations"
    os.makedirs(separations_path, exist_ok=True)

    model, confidence_threshold, nms_threshold, image_size, gpu = figure_separator_model
    # print("This is the model loaded:   ",model)
    model.eval()

    ## Runs model on each image 

    # OLD
    # image = plt.imread(figure_path, "jpg")
    # if len(image.shape) == 2:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # elif image.shape[2] == 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # image, info_image = preprocess(image, image_size, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    # image = np.transpose(image / 255., (2, 0, 1))
    # image = torch.from_numpy(image).float().unsqueeze(0)
    # if gpu > 0:
    #     image = Variable(image.type(torch.cuda.FloatTensor))
    # else:
    #     image = Variable(image.type(torch.FloatTensor))

    # print("\n\n\n\n")
    # print("FIG PATH")
    # print(figure_path)

    img = cv2.imread(figure_path)
    img_raw = img.copy()[:, :, ::-1].transpose((2, 0, 1))
    img, info_image = preprocess(img, image_size, jitter=0)  # info = (h, w, nh, nw, dx, dy)
    img = np.transpose(img / 255., (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)

    gpu = -1
    if gpu >= 0:
        model.cuda(gpu)
        image = Variable(img.type(torch.cuda.FloatTensor))
    else:
        image = Variable(img.type(torch.FloatTensor))

    with torch.no_grad():
        outputs = model(image)
        outputs = postprocess(outputs, 80, confidence_threshold, nms_threshold)
   
    object_data = (outputs, info_image)

    if outputs[0] is None:
        # print("\nNo Objects Detected!!")
        outputs = [[]]
        #continue

    bboxes = list()
    classes = list()
    colors = list()
    confidences = []
        
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
        cls_id = coco_class_ids[int(cls_pred)]
#             print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
#             print('\t+ Label: %s, Conf: %.5f' %
#                   (coco_class_names[cls_id], cls_conf.item()))
        box = yolobox2label([y1, x1, y2, x2], info_image)
        bboxes.append(box)
        classes.append(cls_id)
        colors.append(coco_class_colors[int(cls_pred)])
        confidences.append("%.3f"%(cls_conf.item()))

    
    for i in range(len(bboxes)):
        y1,x1,y2,x2 = bboxes[i]
        bboxes[i] = [int(x1.data.cpu().numpy()),int(y1.data.cpu().numpy()),int(x2.data.cpu().numpy()),int(y2.data.cpu().numpy())]

    bboxes,classes,confidences = NonMaximumSuppressionOnPosition (bboxes,classes,confidences,NMS_threshold=0.2,target=["subfigure_label","master_image"])
    
    result_record = SavingResults(bboxes, classes, confidences, figure_path, \
                                  coco_class_names,separations_path,  \
                                  False, None, json_dict)
    
    json_dict = result_record.json_dict
    

    return json_dict
    # return write_object_dictionary(object_data)

# -------------------------------- #  
# -------- TEMPORARY HOME -------- #
# -------------------------------- #  
def get_figure_images(figure_dict=dict, figure_path=str, search_query=dict) -> None:
    """
    Extract and save patches from figure based on bbox information in the figure_dict

    Args:
        figure_dict: A dictionary with classified image_objects extracted from figure
        figure_path: A path to the figure to separate
        search_query: A query json

    Returns:
        None
    """
    def labelbox_to_patch(lb,img):
        x1,y1=lb[0]["x"],lb[0]["y"]
        x2,y2=lb[2]["x"],lb[2]["y"]
        return img[y1:y2,x1:x2]

    figure_root = ".".join(figure_path.split('/')[-1].split('.')[0:-1])
    figure_ext = ".png"

    os.makedirs(search_query['results_dir'] + "/images/", exist_ok=True)
    
    count = 1
    image = plt.imread(figure_path)
    for a in figure_dict['master_images']:
        patch = labelbox_to_patch(a["geometry"],image)
        plt.imsave(search_query['results_dir']+"/images/"+figure_root+"_"+str(count).zfill(2)+figure_ext,patch)
        count +=1
