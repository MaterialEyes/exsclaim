import matplotlib.pyplot as plt


def format_classification_results(results_file):
    """ Plot training and testing loss and accuracy of a classifier 

    Args:
        results_file (string): Name of a file in results/ directory
            containing results formated as:
            Epoch X/Y.. Test Loss: i.. Train Loss: j.. Test Accuracy: k\n
    """
    with open("results/{}".format(results_file), "r") as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if "Epoch" in line]

    epochs_seen = set()
    epochs = []
    test_losses = []
    train_losses = []
    accuracies = []
    for line in lines:
        epoch_frag, train_frag, test_frag, accuracy_frag = line.split("..")
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

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax1.plot(epochs, train_losses, label="Training Loss", color="tab:red")
    ax1.plot(epochs, test_losses, label="Testing Loss", color="tab:purple")
    
    plt.title(results_file.split(".")[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("% Accuracy")
    ax2.plot(epochs, accuracies, label="Test Accuracy", color="tab:green")

    fig.legend()
    fig.tight_layout()

    plt.show()


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
    print(precision)
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
    #format_classification_results("scale_reader_resnet152.txt")
    format_object_detection_results("scale_bar_detector.txt")
