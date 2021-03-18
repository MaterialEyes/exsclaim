/**
 * @param {string} url - The source image
 * @param {number} aspectRatio - The aspect ratio
 * @return {Promise<HTMLCanvasElement>} A Promise that resolves with the resulting image as a canvas element
 */
function crop(subfigure_image) {
  // extract parameters stored in html canvas element
  // note: these are passed as attributes to allow the django template
  // language to populate them (which can only happen in html) while
  // avoiding inline scripts (which are unsafe)
  var url = subfigure_image.getAttribute("url");
  var x1 = subfigure_image.getAttribute("x1");
  var y1 = subfigure_image.getAttribute("y1");
  var x2 = subfigure_image.getAttribute("x2");
  var y2 = subfigure_image.getAttribute("y2");
  var height = y2 - y1;
  var width = x2 - x1;
  subfigure_image.setAttribute("width", width);
  subfigure_image.setAttribute("height", height);

  // create a new image to load the content
  var image = new Image(),
  ctx = subfigure_image.getContext('2d');
  image.src = url;

  image.onload = function(){
    ctx.drawImage(image,
        x1, y1,   // Start at 70/20 pixels from the left and the top of the image (crop),
        width, height,   // "Get" a `50 * 50` (w * h) area from the source image (crop),
        0, 0,     // Place the result at 0, 0 in the canvas,
        width, height); // With as width / height: 100 * 100 (scale)
  }
}

function resize(column) {
  total_columns = column.getAttribute("columns");
  var width = 100 / total_columns;
  column.style.flex = String(width) + "%";
}

document.addEventListener("DOMContentLoaded", function(e) {
  // Crop Figures to display subfigures
  var subfigure_images = document.getElementsByClassName("subfigure_image");
  Array.from(subfigure_images).forEach(crop);

  // Show scale confidence slider value
  var scale_slider = document.getElementById("scale_confidence");
  var scale_slider_value = document.getElementById("scale_confidence_value");
  scale_slider_value.innerHTML = scale_slider.value / 100; 
  scale_slider.oninput = function() {
    scale_slider_value.innerHTML = this.value / 100;
  }

  // For Grid View: size the number of columns
  var columns = document.getElementsByClassName("column");
  console.log(columns);
  Array.from(columns).forEach(resize);

});
