classifier_model=exsclaim/figures/checkpoints/classifier_model.pt
dir=exsclaim/figures/checkpoints
classifier_googleid="1ZGkbAXeobzx4qFB7JfdhwYfwbmsaDJns"
if [ ! -f "$classifier_model" ]; then
    if [ ! -d "$dir" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${classifier_googleid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${classifier_googleid}" -o ${classifier_model}
else
    echo "Already downloaded classifier_model.pt"
fi

object_model=exsclaim/figures/checkpoints/object_detection_model.pt
object_googleid="1cajTOMsompg-Q8rF3zeby6LNaP_0tpCE"
if [ ! -f "$object_model" ]; then
    if [ ! -d "$dir" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${object_googleid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${object_googleid}" -o ${object_model}
fi

text_model=exsclaim/figures/checkpoints/text_recognition_model.pt
text_googleid="1frtRxgCs8wQsyGRqh6NQOFWaZ897GLs8"
if [ ! -f "$text_model" ]; then
    if [ ! -d "$dir" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${text_googleid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${text_googleid}" -o ${text_model}
else
    echo "Already downloaded text_recognition_model.pt"
fi

scale_detection=exsclaim/figures/checkpoints/scale_bar_detection_model.pt
scale_detection_googleid="18jGI7EsTJEYpZt2ISlFYaqfQZT4ttab5"
if [ ! -f "$scale_detection" ]; then
    if [ ! -d "$dir" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${scale_detection_googleid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${scale_detection_googleid}" -o ${scale_detection}
else
    echo "Already downloaded scale_bar_detection_model.pt"
fi