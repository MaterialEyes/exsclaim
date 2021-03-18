function updateProgress(progressBarElement, progressBarMessageElement, progress) {
    progressBarElement.style.width = progress.percent + "%";
    progressBarMessageElement.innerHTML = progress.current + ' of ' + progress.total + ' processed.';
    console.log(progress);
}

document.addEventListener("DOMContentLoaded", function(e) {

    var trigger = document.getElementById('submit_query');
    trigger.addEventListener('click', function(e) {
        var bar = document.getElementById("progress_bar");
        var barMessage = document.getElementById("progress_message");
        for (var i = 0; i < 11; i++) {
            console.log("someting");
            setTimeout(updateProgress, 5000 * i, bar, barMessage, {
                percent: 10 * i,
                current: 10 * i,
                total: 100
            })
        }
    })
})