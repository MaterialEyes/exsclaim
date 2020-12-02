import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import json
import matplotlib


def format_classification_results(results_file, ax=None):
    """ Plot training and testing loss and accuracy of a classifier 

    Args:
        results_file (string): Name of a file in results/ directory
            containing results formated as:
            Epoch X/Y.. Test Loss: i.. Train Loss: j.. Test Accuracy: k\n
    """
    if "-" in results_file:
        results_file = "-".join(results_file.split("-")[:-1]) + ".txt"

    with open("{}".format(results_file), "r") as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if "Epoch" in line]

    epochs_seen = set()
    epochs = []
    test_losses = []
    train_losses = []
    accuracies = []
    for line in lines:
        epoch_frag, train_frag, test_frag, accuracy_frag, other = line.split("..")
        epoch = int(epoch_frag.split(" ")[-1].split("/")[0])
        if epoch in epochs_seen:
            continue
        training_loss = float(train_frag.split(" ")[-1])
        test_loss = float(test_frag.split(" ")[-1])
        accuracy = float(accuracy_frag.split(" ")[-1])
        epochs_seen.add(epoch)
        epochs.append(epoch)
        test_losses.append(test_loss)
        train_losses.append(training_loss)
        accuracies.append(accuracy)       

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylim([0.0, 5.0])

    line_1, = ax.plot(epochs, train_losses, label="Training Loss", color="tab:red")
    line_2, = ax.plot(epochs, test_losses, label="Testing Loss", color="tab:purple")
    
    ax.title.set_text("Training Results")

    ax2 = ax.twinx()
    ax2.set_ylabel("% Accuracy")
    ax2.set_ylim([0.0, 1.0])
    line_3, = ax2.plot(epochs, accuracies, label="Test Accuracy", color="tab:green")
    plt.legend([line_1, line_2, line_3], ["Training Loss", "Testing Loss", "Test Accuracy"])


def make_confusion_matrix(model_name, actual, predicted, ax=None, pdf=None):
    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predicted, name='Predicted')
    confusion_matrix = pd.crosstab(y_actu, y_pred)
    idx_to_class = get_idx_to_class(model_name)

    # format subplot
    ax.title.set_text("Confusion Matrix")
    cmap = plt.cm.gray_r
    ax_image = ax.matshow(confusion_matrix, cmap=cmap) # imshow
    plt.colorbar(ax_image, ax=ax)
    x_tick_marks = np.arange(len(confusion_matrix.columns))
    y_tick_marks = np.arange(len(confusion_matrix.index))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(confusion_matrix.columns)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(confusion_matrix.index)
    #plt.tight_layout()
    ax.set_ylabel(confusion_matrix.index.name)
    ax.set_xlabel(confusion_matrix.columns.name)
    #plt.show()


def get_idx_to_class(model_name):
    ## Code to set up scale label reading model(s)
    # id_to_class dictionaries for model outputs
    all = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '0.6 A', 16: '0.6 nm', 17: '0.6 um', 18: '0.7 A', 19: '0.7 nm', 20: '0.7 um', 21: '0.8 A', 22: '0.8 nm', 23: '0.8 um', 24: '0.9 A', 25: '0.9 nm', 26: '0.9 um', 27: '1 A', 28: '1 nm', 29: '1 um', 30: '10 A', 31: '10 nm', 32: '10 um', 33: '100 A', 34: '100 nm', 35: '100 um', 36: '2 A', 37: '2 nm', 38: '2 um', 39: '2.5 A', 40: '2.5 nm', 41: '2.5 um', 42: '20 A', 43: '20 nm', 44: '20 um', 45: '200 A', 46: '200 nm', 47: '200 um', 48: '25 A', 49: '25 nm', 50: '25 um', 51: '250 A', 52: '250 nm', 53: '250 um', 54: '3 A', 55: '3 nm', 56: '3 um', 57: '30 A', 58: '30 nm', 59: '30 um', 60: '300 A', 61: '300 nm', 62: '300 um', 63: '4 A', 64: '4 nm', 65: '4 um', 66: '40 A', 67: '40 nm', 68: '40 um', 69: '400 A', 70: '400 nm', 71: '400 um', 72: '5 A', 73: '5 nm', 74: '5 um', 75: '50 A', 76: '50 nm', 77: '50 um', 78: '500 A', 79: '500 nm', 80: '500 um', 81: '6 A', 82: '6 nm', 83: '6 um', 84: '60 A', 85: '60 nm', 86: '60 um', 87: '600 A', 88: '600 nm', 89: '600 um', 90: '7 A', 91: '7 nm', 92: '7 um', 93: '70 A', 94: '70 nm', 95: '70 um', 96: '700 A', 97: '700 nm', 98: '700 um', 99: '8 A', 100: '8 nm', 101: '8 um', 102: '80 A', 103: '80 nm', 104: '80 um', 105: '800 A', 106: '800 nm', 107: '800 um', 108: '9 A', 109: '9 nm', 110: '9 um', 111: '90 A', 112: '90 nm', 113: '90 um', 114: '900 A', 115: '900 nm', 116: '900 um'}
    some = {0: '0.1 A', 1: '0.1 nm', 2: '0.1 um', 3: '0.2 A', 4: '0.2 nm', 5: '0.2 um', 6: '0.3 A', 7: '0.3 nm', 8: '0.3 um', 9: '0.4 A', 10: '0.4 nm', 11: '0.4 um', 12: '0.5 A', 13: '0.5 nm', 14: '0.5 um', 15: '1 A', 16: '1 nm', 17: '1 um', 18: '10 A', 19: '10 nm', 20: '10 um', 21: '100 A', 22: '100 nm', 23: '100 um', 24: '2 A', 25: '2 nm', 26: '2 um', 27: '2.5 A', 28: '2.5 nm', 29: '2.5 um', 30: '20 A', 31: '20 nm', 32: '20 um', 33: '200 A', 34: '200 nm', 35: '200 um', 36: '25 A', 37: '25 nm', 38: '25 um', 39: '250 A', 40: '250 nm', 41: '250 um', 42: '3 A', 43: '3 nm', 44: '3 um', 45: '30 A', 46: '30 nm', 47: '30 um', 48: '300 A', 49: '300 nm', 50: '300 um', 51: '4 A', 52: '4 nm', 53: '4 um', 54: '40 A', 55: '40 nm', 56: '40 um', 57: '400 A', 58: '400 nm', 59: '400 um', 60: '5 A', 61: '5 nm', 62: '5 um', 63: '50 A', 64: '50 nm', 65: '50 um', 66: '500 A', 67: '500 nm', 68: '500 um'}        
    scale_some = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '1', 6: '10', 7: '100', 8: '2', 9: '2.5', 10: '20', 11: '200', 12: '25', 13: '250', 14: '3', 15: '30', 16: '300', 17: '4', 18: '40', 19: '400', 20: '5', 21: '50', 22: '500'}
    scale_all = {0: '0.1', 1: '0.2', 2: '0.3', 3: '0.4', 4: '0.5', 5: '0.6', 6: '0.7', 7: '0.8', 8: '0.9', 9: '1', 10: '10', 11: '100', 12: '2', 13: '2.5', 14: '20', 15: '200', 16: '25', 17: '250', 18: '3', 19: '30', 20: '300', 21: '4', 22: '40', 23: '400', 24: '5', 25: '50', 26: '500', 27: '6', 28: '60', 29: '600', 30: '7', 31: '70', 32: '700', 33: '8', 34: '80', 35: '800', 36: '9', 37: '90', 38: '900'}
    unit_data = {0: 'A', 1: 'mm', 2: 'nm', 3: 'um'}
    dataset_to_dict = {"all" : all, "some": some, "scale_all": scale_all, "scale_some": scale_some, "unit_data": unit_data}
    dataset_name = "_".join(model_name.split("-")[0].split("_")[:-1])
    idx_to_class = dataset_to_dict[dataset_name]
    return idx_to_class

def get_accuracy_stats(model_name, results_dict):
    idx_to_class = get_idx_to_class(model_name)
    actual = results_dict["actual_idx"]
    predicted = results_dict["predicted_idx"]
    total = len(predicted)
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    classes_stats = {}
    for class_idx in idx_to_class:
        false_positives = float(len([i for i in range(total) if (predicted[i] == class_idx and actual[i] != class_idx)])) 
        false_negatives = float(len([i for i in range(total) if (predicted[i] != class_idx and actual[i] == class_idx)]))
        true_positives =  float(len([i for i in range(total) if (predicted[i] == class_idx and actual[i] == class_idx)]))
        true_negatives = total - false_negatives - false_positives - true_positives
        total_fn += false_negatives
        total_fp += false_positives
        total_tn += true_negatives
        total_tp += true_positives
        class_name = idx_to_class[class_idx]
        class_stats = {"Recall":    true_positives / (true_positives + false_negatives + 0.0000001),
                       "Precision": true_positives / (true_positives + false_positives + 0.0000001)
                      }
        classes_stats[class_name] = class_stats
    classes_stats["all"] = {"Recall": total_tp / (total_tp + total_fn),
                            "Precision": total_tp / (total_tp + total_fp)}
    return classes_stats

def make_metadata_chart(model_name, results_dict, ax=None):
    ax.axis("off")
    model_name = model_name.split(".")[0]
    model, epochs = model_name.split("-")
    dataset = "_".join(model.split("_")[:-1])
    depth = model.split("_")[-1]

    classes_stats = get_accuracy_stats(model_name, results_dict)
    total_stats = classes_stats["all"]
    accuracy = results_dict["accuracy"]
    ax.text(0, 0, "Dataset: {}\nDepth: {}\nEpochs: {}\nAccuracy: {}".format(dataset, depth, epochs, str(accuracy)))


def get_accuracy_by_confidence(resutls_dict, ax=None):
    actual = resutls_dict["actual_idx"]
    predicted = resutls_dict["predicted_idx"]
    confidences = resutls_dict["confidence"]

    confidence_thresholds = [0.05*i for i in range(1, 20)]
    accuracies = []
    valid = []
    for threshold in confidence_thresholds:
        correct = 0
        incorrect = 0
        num_valid = 0
        for prediction, truth, confidence in zip(predicted, actual, confidences):
            if confidence < threshold:
                continue
            num_valid += 1
            if prediction == truth:
                correct += 1
            else:
                incorrect += 1
        accuracies.append(correct / float(correct + incorrect + 0.00000001))
        valid.append(num_valid)
    
    ax.plot(confidence_thresholds, accuracies)
    ax.title.set_text("Accuracy By Confidence Threshold")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    
    ax2 = ax.twinx()
    ax2.set_ylabel("# of Images")
    #ax2.set_ylim([0.0, 1.0])
    line_3 = ax2.bar(confidence_thresholds, valid, width=0.01, align='center', label="Test Accuracy", color="tab:green")



def generate_report(results_file, training_results_directory):
    with open("results.txt", "r") as f:
        results_dict = json.load(f)
    
    ## Configure matplotlib
    font = {'size'   : 4}
    matplotlib.rc('font', **font)

    with PdfPages('multipage_pdf.pdf') as pdf:
        for model_name in results_dict:
                #fig = plt.figure()
                plt.title(model_name)
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                fig.suptitle(model_name)
                model_dict = results_dict[model_name]
                make_confusion_matrix(model_name, model_dict["actual_class"],
                             model_dict["predicted_class"], ax=ax1)
                get_accuracy_by_confidence(model_dict, ax=ax2)

                results_file = training_results_directory + model_name
                format_classification_results(results_file, ax=ax3)
                make_metadata_chart(model_name, model_dict, ax=ax4)

                plt.tight_layout(pad=1.5)

                pdf.savefig()
                plt.close()



def format_object_detection_results(results_file):
    """ Plots precision and recall scores for object detector

    Args:
        results_file (string): Name of a file in results/ directory
    """   
    with open("results/{}".format(results_file), "r") as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines]
    mAP = [float(line.split("=")[-1].strip()) for line in lines if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line]
    mAP_50 = [float(line.split("=")[-1].strip()) for line in lines if "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in line]
    mAP_75 = [float(line.split("=")[-1].strip()) for line in lines if "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]" in line]
    precision = [float(line.split("=")[-1].strip()) for line in lines if "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line]
    epochs = [i for i in range(len(mAP))]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")

    ax1.plot(epochs, mAP, label="AP @ [.5:.95]", color="tab:red")
    ax1.plot(epochs, mAP_50, label="AP @ 0.5", color="tab:purple")
    ax1.plot(epochs, mAP_75, label="AP @ 0.75", color="tab:green")
    ax1.plot(epochs, precision, label="AR @ [.5:.95]", color="tab:blue")
    plt.title(results_file.split(".")[0])

    fig.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    #format_classification_results("/home/trevor/Documents/argonne/exsclaim/results/pretrained/scale_all_50.txt")
    #format_object_detection_results("scale_bar_detector.txt")
    generate_report("results.txt", "/home/tspread/exsclaim/exsclaim/figures/scale/results/pretrained/")