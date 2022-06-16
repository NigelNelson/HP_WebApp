// Class: SE2840 - Menu Filter
// Name: YOUR NAME HERE
// Class Section: N/A
class App extends React.Component {
  constructor(props) {
    super(props); // Set state as needed

    this.state = {
      hist_src: null,
      mri_src: null,
      is_hist_loaded: false,
      is_mri_loaded: true,
      mri_data: null,
      hist_data: null
    };
  }

  render() {
    const hist_id = 'hist_img';
    const mri_id = 'mri_img';
    const hist_canvas_id = 'hist_canvas';
    const mri_canvas_id = 'mri_canvas';
    let hist_data = this.state.hist_data;
    let mri_data = this.state.mri_data;
    var mri_canvas = document.getElementById(mri_canvas_id);
    var hist_canvas = document.getElementById(hist_canvas_id);
    var mri_shapes = [];
    var hist_shapes = [];

    function readNIFTI(name, data) {
      var mri_div = document.getElementById("mri_div");
      var canvas = document.createElement("canvas");
      canvas.id = mri_canvas_id;
      mri_div.append(canvas); // var canvas = document.getElementById(mri_canvas_id);

      var niftiHeader, niftiImage; // parse nifti

      if (nifti.isCompressed(data)) {
        data = nifti.decompress(data);
      }

      if (nifti.isNIFTI(data)) {
        niftiHeader = nifti.readHeader(data);
        niftiImage = nifti.readImage(niftiHeader, data);
      } // draw slice


      drawCanvas(canvas, 0, niftiHeader, niftiImage);
    }

    function drawCanvas(canvas, slice, niftiHeader, niftiImage) {
      // get nifti dimensions
      var cols = niftiHeader.dims[1];
      var rows = niftiHeader.dims[2]; // set canvas dimensions to nifti slice dimensions

      canvas.width = cols;
      canvas.height = rows; // make canvas image data

      var ctx = canvas.getContext("2d");
      var canvasImageData = ctx.createImageData(canvas.width, canvas.height); // convert raw data to typed array based on nifti datatype

      var typedData;

      if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
        typedData = new Uint8Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
        typedData = new Int16Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32) {
        typedData = new Int32Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
        typedData = new Float32Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) {
        typedData = new Float64Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT8) {
        typedData = new Int8Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT16) {
        typedData = new Uint16Array(niftiImage);
      } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT32) {
        typedData = new Uint32Array(niftiImage);
      } else {
        return;
      } // offset to specified slice


      var sliceSize = cols * rows;
      var sliceOffset = sliceSize * 0; // draw pixels

      for (var row = 0; row < rows; row++) {
        var rowOffset = row * cols;

        for (var col = 0; col < cols; col++) {
          var offset = sliceOffset + rowOffset + col;
          var value = typedData[offset];
          var ratio = 255 / 32768;
          value *= ratio;
          /*
             Assumes data is 8-bit, otherwise you would need to first convert
             to 0-255 range based on datatype range, data range (iterate through
             data to find), or display range (cal_min/max).
               Other things to take into consideration:
               - data scale: scl_slope and scl_inter, apply to raw value before
                 applying display range
               - orientation: displays in raw orientation, see nifti orientation
                 info for how to orient data
               - assumes voxel shape (pixDims) is isometric, if not, you'll need
                 to apply transform to the canvas
               - byte order: see littleEndian flag
          */

          canvasImageData.data[(rowOffset + col) * 4] = value & 0xFF;
          canvasImageData.data[(rowOffset + col) * 4 + 1] = value & 0xFF;
          canvasImageData.data[(rowOffset + col) * 4 + 2] = value & 0xFF;
          canvasImageData.data[(rowOffset + col) * 4 + 3] = 0xFF;
        }
      }

      ctx.putImageData(canvasImageData, 0, 0);
      var tempCanvas = document.createElement("canvas");
      var tempCtx = tempCanvas.getContext("2d");
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate(90 * Math.PI / 180);
      ctx.translate(-canvas.width / 2, -canvas.height / 2);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(1, -1);
      ctx.drawImage(tempCanvas, 0, 0, tempCanvas.width, -tempCanvas.height);
      ctx.restore();
      mri_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
      log_mri_data(mri_data);
    }

    function makeSlice(file, start, length) {
      var fileType = typeof File;

      if (fileType === 'undefined') {
        return function () {};
      }

      if (File.prototype.slice) {
        return file.slice(start, start + length);
      }

      if (File.prototype.mozSlice) {
        return file.mozSlice(start, length);
      }

      if (File.prototype.webkitSlice) {
        return file.webkitSlice(start, length);
      }

      return null;
    }

    function readFile(file) {
      var blob = makeSlice(file, 0, file.size);
      var reader = new FileReader();

      reader.onloadend = function (evt) {
        if (evt.target.readyState === FileReader.DONE) {
          readNIFTI(file.name, evt.target.result);
        }
      };

      reader.readAsArrayBuffer(blob);
    }

    const show_hist = () => {
      var c = document.getElementById(hist_id);
      var xhr = new XMLHttpRequest();
      xhr.responseType = 'arraybuffer';
      xhr.open('GET', URL.createObjectURL(event.target.files[0]));

      xhr.onload = function (e) {
        var tiff = new Tiff({
          buffer: xhr.response
        });
        var canvas = tiff.toCanvas();
        canvas.id = hist_canvas_id;
        c.append(canvas);
        let ctx = canvas.getContext("2d");
        hist_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        log_hist_data(hist_data);
      };

      xhr.send();
      this.setState({
        is_hist_loaded: true
      });
    };

    const log_mri_data = data => {
      this.setState({
        mri_data: data
      });
    };

    const log_hist_data = data => {
      this.setState({
        hist_data: data
      });
    };

    const show_mri = () => {
      var files = event.target.files;
      readFile(files[0]);
      this.setState({
        is_mri_loaded: true
      });
    };

    const drawCircle = (ctx, x, y, radius, fill, stroke, strokeWidth) => {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI, false);

      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }

      if (stroke) {
        ctx.lineWidth = strokeWidth;
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
    };

    const display_points = () => {
      let hist_canvas = document.getElementById(hist_canvas_id);
      let hist_context = hist_canvas.getContext('2d');
      let mri_canvas = document.getElementById(mri_canvas_id);
      let mri_context = mri_canvas.getContext('2d');
      var files = event.target.files;
      var reader = new FileReader();
      reader.readAsText(files[0]);

      reader.onload = function (event) {
        var csv = event.target.result;
        var data = $.csv.toArrays(csv);
        data.forEach(function (points) {
          let hist_x = Number(points[0]);
          let hist_y = Number(points[1]);
          let mri_x = Number(points[2]);
          let mri_y = Number(points[3]);
          let radius = 7;
          let strokeWidth = 2;
          let fill = 'black';
          let stroke = 'red';
          drawCircle(hist_context, hist_x, hist_y, radius, fill, stroke, strokeWidth);
          drawCircle(mri_context, mri_x, mri_y, radius, fill, stroke, strokeWidth);
          hist_shapes.push({
            x: hist_x,
            y: hist_y,
            radius: radius,
            fill: fill,
            stroke: strokeWidth,
            strokeWith: strokeWidth
          });
          mri_shapes.push({
            x: mri_x,
            y: mri_y,
            radius: radius,
            fill: fill,
            stroke: stroke,
            strokeWith: strokeWidth
          });
        });
      };
    };

    const download_points = () => {
      var csv = "";

      for (let i = 0; i < hist_shapes.length; i++) {
        csv += [hist_shapes[i].x, hist_shapes[i].y, mri_shapes[i].x, mri_shapes[i].y].join(',');
        csv += "\n";
      }

      var hiddenElement = document.createElement('a');
      hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
      hiddenElement.target = '_blank';
      hiddenElement.download = 'hist+mri_points.csv';
      hiddenElement.click();
    }; // given mouse X & Y (mx & my) and shape object
    // return true/false whether mouse is inside the shape


    function isMouseInShape(mx, my, shape) {
      // this is a circle
      var dx = mx - shape.x;
      var dy = my - shape.y; // math test to see if mouse is inside circle

      if (dx * dx + dy * dy < shape.radius * shape.radius) {
        // yes, mouse is inside this circle
        return true;
      } // the mouse isn't in any of the shapes


      return false;
    }

    function getMousePos(canvas, evt) {
      var rect = canvas.getBoundingClientRect();
      return {
        x: (evt.clientX - rect.left) / (rect.right - rect.left) * canvas.width,
        y: (evt.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height
      };
    }

    function handleMouseDown(e) {
      // tell the browser we're handling this event
      e.preventDefault();
      e.stopPropagation();
      let mousePos = getMousePos(this, e);
      let shapes;
      startX = parseInt(mousePos.x);
      startY = parseInt(mousePos.y); // calculate the current mouse position
      // startX = parseInt(e.clientX - offsetX);
      // startY = parseInt(e.clientY - offsetY);
      // test mouse position against all shapes
      // post result if mouse is in a shape

      if (this.id === mri_canvas_id) {
        shapes = mri_shapes;
      } else {
        shapes = hist_shapes;
      }

      for (var i = 0; i < shapes.length; i++) {
        if (isMouseInShape(startX, startY, shapes[i])) {
          // the mouse is inside this shape
          // select this shape
          selectedShapeIndex = i; // set the isDragging flag

          isDragging = true; // and return (==stop looking for
          //     further shapes under the mouse)

          return;
        }
      }
    }

    function handleMouseUp(e) {
      // return if we're not dragging
      if (!isDragging) {
        return;
      } // tell the browser we're handling this event


      e.preventDefault();
      e.stopPropagation(); // the drag is over -- clear the isDragging flag

      isDragging = false;
    }

    function handleMouseOut(e) {
      // return if we're not dragging
      if (!isDragging) {
        return;
      } // tell the browser we're handling this event


      e.preventDefault();
      e.stopPropagation(); // the drag is over -- clear the isDragging flag

      isDragging = false;
    }

    function handleMouseMove(e) {
      // return if we're not dragging
      if (!isDragging) {
        return;
      } // tell the browser we're handling this event


      e.preventDefault();
      e.stopPropagation(); // calculate the current mouse position

      let mousePos = getMousePos(this, e);
      let mouseX = parseInt(mousePos.x);
      let mouseY = parseInt(mousePos.y); // let mouseX = parseInt(e.clientX - offsetX);
      // let mouseY = parseInt(e.clientY - offsetY);
      // how far has the mouse dragged from its previous mousemove position?

      var dx = mouseX - startX;
      var dy = mouseY - startY; // move the selected shape by the drag distance

      let shapes;

      if (this.id === mri_canvas_id) {
        shapes = mri_shapes;
      } else {
        shapes = hist_shapes;
      }

      var selectedShape = shapes[selectedShapeIndex];
      selectedShape.x += dx;
      selectedShape.y += dy; // clear the canvas and redraw all shapes

      drawAll(this); // update the starting drag position (== the current mouse position)

      startX = mouseX;
      startY = mouseY;
    } // clear the canvas and
    // redraw all shapes in their current positions


    function drawAll(canvas) {
      let data;
      let shapes;

      if (canvas.id === mri_canvas_id) {
        data = mri_data;
        shapes = mri_shapes;
      } else {
        data = hist_data;
        shapes = hist_shapes;
      }

      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.putImageData(data, 0, 0);

      for (var i = 0; i < shapes.length; i++) {
        var shape = shapes[i];

        if (shape.radius) {
          // it's a circle
          drawCircle(ctx, shape.x, shape.y, shape.radius, shape.fill, shape.stroke, shape.strokeWidth);
        }
      }
    } // drag related vars


    var isDragging = false;
    var startX, startY; // hold the index of the shape being dragged (if any)

    var selectedShapeIndex; // draw the shapes on the canvas

    if (mri_data) {
      drawAll(mri_canvas);
    }

    if (hist_data) {
      drawAll(hist_canvas);
    } // listen for mouse events


    if (mri_data) {
      mri_canvas.onmousedown = handleMouseDown;
      mri_canvas.onmousemove = handleMouseMove;
      mri_canvas.onmouseup = handleMouseUp;
      mri_canvas.onmouseout = handleMouseOut;
    } // if(hist_data){
    //     hist_canvas.onmousedown = handleMouseDown;
    //     hist_canvas.onmousemove = handleMouseMove;
    //     hist_canvas.onmouseup = handleMouseUp;
    //     hist_canvas.onmouseout = handleMouseOut;
    // }
    // Render the application


    return /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail"
    }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h1", null, "LaViolette Points")), /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-body"
    }, /*#__PURE__*/React.createElement("div", {
      className: "nav"
    }), /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-content"
    }, /*#__PURE__*/React.createElement("div", {
      className: "container"
    }, /*#__PURE__*/React.createElement("div", {
      className: "row"
    }, /*#__PURE__*/React.createElement(Image, {
      imgSRC: this.state.hist_src,
      id: hist_id,
      className: "col"
    }), /*#__PURE__*/React.createElement("div", {
      id: "mri_div"
    })), /*#__PURE__*/React.createElement("div", {
      className: "row"
    }, /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: show_hist,
      promptStatement: "Choose a Histology Image:",
      className: "col"
    }), /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: show_mri,
      promptStatement: "Choose an MRI Image:",
      className: "col"
    }), /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: display_points,
      promptStatement: "Choose Histology points:",
      className: "col"
    }), /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-primary",
      onClick: download_points
    }, /*#__PURE__*/React.createElement("svg", {
      xmlns: "http://www.w3.org/2000/svg",
      width: "16",
      height: "16",
      fill: "currentColor",
      className: "bi bi-download m-1",
      viewBox: "0 0 16 16"
    }, /*#__PURE__*/React.createElement("path", {
      d: "M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"
    }), /*#__PURE__*/React.createElement("path", {
      d: "M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"
    })), "Download Points"))))));
  }

}
class Canvas extends React.Component {
  render() {
    let img = document.getElementById(this.props.imgID);

    if (img) {
      return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("canvas", {
        id: this.props.id,
        style: {
          position: 'absolute',
          left: img.offsetLeft + "px",
          top: img.offsetTop + "px"
        }
      }));
    } else {
      return /*#__PURE__*/React.createElement("canvas", {
        id: this.props.id
      });
    }
  }

}
class FileSelect extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("div", {
      className: "mb-3"
    }, /*#__PURE__*/React.createElement("label", {
      htmlFor: "formFile",
      className: "form-label"
    }, this.props.promptStatement), /*#__PURE__*/React.createElement("input", {
      className: "form-control",
      type: "file",
      id: "formFile",
      onChange: event => this.props.onFileSelect(event.target.value)
    }));
  }

}
// import Skeleton from 'react-loading-skeleton'
// import 'react-loading-skeleton/dist/skeleton.css'
class Image extends React.Component {
  render() {
    return (
      /*#__PURE__*/
      // <div>
      //     <img src={this.props.imgSRC} id={this.props.id} width='70%'/>
      // </div>
      React.createElement("div", {
        id: this.props.id
      })
    );
  }

}
// Class: SE2840 - Menu Filter
// Web Application entry point - window.onload

/**
 * Window onload function - Creates the menuItem (unfiltered) array
 *     and renders the application
 */
window.onload = () => {
  ReactDOM.render( /*#__PURE__*/React.createElement(App, null), document.getElementById('root'));
};
class MRICanvas extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("canvas", {
      id: this.props.id
    });
  }

}
