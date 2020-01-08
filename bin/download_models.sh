model_01=exsclaim/figures/checkpoints/snapshot260.ckpt
dir_01=exsclaim/figures/checkpoints
if [ ! -f "$model_01" ]; then
    if [ ! -d "$dir_01" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    gdown -O exsclaim/figures/checkpoints/snapshot260.ckpt https://drive.google.com/uc?id=16vH7FUZXm9hqedqMzne9ToxBajU_hp4i
else
    echo "Already downloaded snapshot260.ckpt"
fi

model_02=exsclaim/figures/checkpoints/snapshot13400.ckpt
dir_02=exsclaim/figures/checkpoints
if [ ! -f "$model_02" ]; then
    if [ ! -d "$dir_02" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    gdown -O exsclaim/figures/checkpoints/snapshot13400.ckpt https://drive.google.com/uc?id=1JXNXiIUrH99y8Zqj02wB0OSw24enUZ3-
    echo "Already downloaded snapshot13400.ckpt"
fi

model_03=exsclaim/figures/checkpoints/snapshot12000.ckpt
dir_03=exsclaim/figures/checkpoints
if [ ! -f "$model_03" ]; then
    if [ ! -d "$dir_03" ] ; then
        mkdir exsclaim/figures/checkpoints
    fi
    gdown -O exsclaim/figures/checkpoints/snapshot12000.ckpt https://drive.google.com/uc?id=1O4a58sQsOSRrpKTkHNP2nqfwbjI4oCj_
else
    echo "Already downloaded snapshot12000.ckpt"
fi

chrome_dir=exsclaim/figures/checkpoints
if [ ! -f "exsclaim/journals/chromedriver" ]; then
    if [ ! -d "chrome_dir" ] ; then
        mkdir exsclaim/journals
    fi
    gdown -O exsclaim/journals/chrometools.zip https://drive.google.com/uc?id=16k_RFZgNCMqIos18zA5dvgW8jFb_JfGA
else
    echo "Already downloaded chrometools!"
fi

cd exsclaim/journals; unzip -q chrometools.zip; mv chrometools/* ../journals ; rm -r chrometools.zip chrometools __MACOSX; cd ../..;