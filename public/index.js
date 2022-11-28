class App extends React.Component {
  constructor(props) {
    super(props); // Set state as needed

    this.state = {
      is_hist_loaded: false,
      is_mri_loaded: true,
      mri_data: null,
      hist_data: null,
      edit_mri_points: false,
      hist_shapes: [],
      mri_shapes: [],
      pred_hist_shapes: [],
      pred_mri_shapes: [],
      display_hist_points: [],
      display_mri_points: [],
      toggle_nums: [],
      current_point_idx: 0,
      hist_file: "",
      mri_file: "",
      hist_mri_points: [],
      sift_points: []
    };
  }

  render() {
    const hist_id = 'hist_img';
    const mri_id = 'mri_img';
    const hist_canvas_id = 'hist_canvas';
    const mri_canvas_id = 'mri_canvas';
    const predict_id = 'predict_btn';
    let hist_mri_points = this.state.hist_mri_points;
    let hist_data = this.state.hist_data;
    let mri_data = this.state.mri_data;
    let hist_file = this.state.hist_file;
    let mri_file = this.state.mri_file;
    let sift_points = this.state.sift_points;
    var mri_canvas = document.getElementById(mri_canvas_id);
    var hist_canvas = document.getElementById(hist_canvas_id);
    var mri_shapes = this.state.mri_shapes;
    var hist_shapes = this.state.hist_shapes;
    let pred_mri_shapes = this.state.pred_mri_shapes;
    let pred_hist_shapes = this.state.pred_hist_shapes;
    var display_hist_points = this.state.display_hist_points;
    var display_mri_points = this.state.display_mri_points;
    var mri_switch_id = "mri_switch";
    var num_toggles = 0;
    var toggle_nums = this.state.toggle_nums;

    const handlePointToggle = num => {
      this.setState({
        display_mri_points: [this.state.mri_shapes[num]],
        display_hist_points: [this.state.hist_shapes[num]],
        current_point_idx: num
      });
    };

    const handleNextPoint = () => {
      let num_points = this.state.toggle_nums.length;
      let point_idx = this.state.current_point_idx + 1;

      if (point_idx < num_points) {
        this.setState({
          display_mri_points: [this.state.mri_shapes[point_idx]],
          display_hist_points: [this.state.hist_shapes[point_idx]],
          current_point_idx: point_idx
        });
      }
    };

    const handlePreviousPoint = () => {
      let point_idx = this.state.current_point_idx - 1;

      if (point_idx >= 0) {
        this.setState({
          display_mri_points: [this.state.mri_shapes[point_idx]],
          display_hist_points: [this.state.hist_shapes[point_idx]],
          current_point_idx: point_idx
        });
      }
    };

    const handleClickALl = () => {
      this.setState({
        display_mri_points: this.state.mri_shapes,
        display_hist_points: this.state.hist_shapes,
        current_point_idx: 0
      });
    };

    const handleClick = () => {
      let mri_switch = document.getElementById(mri_switch_id);

      if (mri_switch.checked) {
        this.setState({
          edit_mri_points: true
        });
      } else {
        this.setState({
          edit_mri_points: false
        });
      }
    };
    /**
     * Equalizes the histogram of an unsigned 1-channel image with values
     * in range [0, 255]. Corresponds to the equalizeHist OpenCV function.
     *
     * @param {Array} src 1-channel source image
     * result is written to src (faster)
     * @return {Array} Destination image
     */


    const equalizeHistogram = src => {
      var srcLength = src.length;
      let dst = src; // Compute histogram and histogram sum:

      var hist = new Float32Array(256);
      var sum = 0;

      for (var i = 0; i < srcLength; ++i) {
        ++hist[~~src[i]];
        ++sum;
      } // Compute integral histogram:


      var prev = hist[0];

      for (var i = 1; i < 256; ++i) {
        prev = hist[i] += prev;
      } // Equalize image:


      var norm = 255 / sum;

      for (var i = 0; i < srcLength; ++i) {
        dst[i] = hist[~~src[i]] * norm;
      }

      return dst;
    };

    function readNIFTI(name, data) {
      var mri_div = document.getElementById(mri_id);
      var canvas = document.createElement("canvas");
      canvas.id = mri_canvas_id;
      canvas.style.margin = "auto";
      canvas.style.width = '100%';
      canvas.style.height = '100%';
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
      } // Perform histogram equalization


      let equalized_arr = new Uint8ClampedArray(equalizeHistogram(canvasImageData.data));
      let equalized_img = new ImageData(equalized_arr, canvasImageData.width, canvasImageData.height);
      ctx.putImageData(equalized_img, 0, 0);
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
      let mri_file = canvas.toDataURL("image/png");
      log_mri_data(mri_data, mri_file);
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
        canvas.style.margin = "auto";
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        c.append(canvas);
        let ctx = canvas.getContext("2d");
        hist_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const byte_img = tiff.toDataURL();
        log_hist_data(hist_data, byte_img);
      };

      xhr.send();
      this.setState({
        is_hist_loaded: true
      });
    };

    const log_mri_data = (data, file) => {
      this.setState({
        mri_data: data,
        mri_file: file
      });
    };

    const log_points = data => {
      this.setState({
        hist_mri_points: data
      });
    };

    const log_sift_points = data => {
      this.setState({
        sift_points: data
      });
    };

    const log_hist_data = (data, byte_img) => {
      this.setState({
        hist_data: data,
        hist_file: byte_img
      });
    };

    const log_toggle_nums = toggle_nums => {
      this.setState({
        toggle_nums: toggle_nums
      });
    };

    const log_display_points = (hist_points, mri_points) => {
      this.setState({
        display_hist_points: hist_points,
        display_mri_points: mri_points
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
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.beginPath();
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);

      if (fill) {
        ctx.fill();
      }

      ctx.arc(x, y, radius + 2, 0, 2 * Math.PI, false);

      if (stroke) {
        ctx.lineWidth = strokeWidth;
        ctx.stroke();
      }
    };

    const selectColor = number => {
      const hue = number * 137.508; // use golden angle approximation

      return `hsl(${hue},100%,50%)`;
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

        if (sift_points.length < 1) {
          data.forEach(function (points) {
            let hist_x = Number(points[1]);
            let hist_y = Number(points[0]);
            let mri_x = Number(points[3]);
            let mri_y = Number(points[2]);
            let radius = 7;
            let strokeWidth = 2;
            let fill = 'red';
            let stroke = 'red';
            let color = selectColor(num_toggles);
            drawCircle(hist_context, hist_x, hist_y, radius, color, color, strokeWidth);
            drawCircle(mri_context, mri_x, mri_y, radius, color, color, strokeWidth);
            hist_shapes.push({
              x: hist_x,
              y: hist_y,
              radius: radius,
              fill: color,
              stroke: color,
              strokeWith: strokeWidth
            });
            mri_shapes.push({
              x: mri_x,
              y: mri_y,
              radius: radius,
              fill: color,
              stroke: color,
              strokeWith: strokeWidth
            });
            toggle_nums.push(num_toggles);
            num_toggles = num_toggles + 1;
          });
        }

        display_mri_points = mri_shapes;
        display_hist_points = hist_shapes;
        log_points(data);
        log_toggle_nums(toggle_nums);
        log_display_points(display_hist_points, display_mri_points);
      };

      this.setState({
        hist_shapes: hist_shapes,
        mri_shapes: mri_shapes,
        toggle_nums: toggle_nums
      });
    };

    const download_points = () => {
      var csv = "";

      for (let i = 0; i < hist_shapes.length; i++) {
        csv += [hist_shapes[i].y, hist_shapes[i].x, mri_shapes[i].y, mri_shapes[i].x].join(',');
        csv += "\n";
      }

      var hiddenElement = document.createElement('a');
      hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
      hiddenElement.target = '_blank';
      hiddenElement.download = 'corrected_histmri_points.csv';
      hiddenElement.click();
    }; //////////////////// Functions Enabling Dragging and Dropping of Points //////////////////
    // given mouse X & Y (mx & my) and shape object
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
        shapes = display_mri_points;
      } else {
        data = hist_data;
        shapes = display_hist_points;
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
      if (this.state.edit_mri_points) {
        mri_canvas.onmousedown = handleMouseDown;
        mri_canvas.onmousemove = handleMouseMove;
        mri_canvas.onmouseup = handleMouseUp;
        mri_canvas.onmouseout = handleMouseOut;
      } else {
        mri_canvas.onmousedown = null;
        mri_canvas.onmousemove = null;
        mri_canvas.onmouseup = null;
        mri_canvas.onmouseout = null;
      }
    } //////////////////////////// Helper Functions For Python Scipt Responses  //////////////////////////


    const display_sift_hist_points = data => {
      let mri_canvas = document.getElementById(mri_canvas_id);
      let mri_context = mri_canvas.getContext('2d');
      mri_shapes = [];
      num_toggles = 0;
      data.forEach(function (points) {
        let mri_x = Number(points[1]);
        let mri_y = Number(points[0]);
        let radius = 7;
        let strokeWidth = 2;
        let fill = 'red';
        let stroke = 'red';
        let color = selectColor(num_toggles);
        drawCircle(mri_context, mri_x, mri_y, radius, color, color, strokeWidth);
        mri_shapes.push({
          x: mri_x,
          y: mri_y,
          radius: radius,
          fill: color,
          stroke: color,
          strokeWith: strokeWidth
        });
        toggle_nums.push(num_toggles);
        num_toggles = num_toggles + 1;
      });
      display_mri_points = mri_shapes;
      log_toggle_nums(toggle_nums);
      log_display_points(display_hist_points, display_mri_points);
      this.setState({
        mri_shapes: mri_shapes,
        toggle_nums: toggle_nums
      });
    };

    const display_predicted_mri_points = data => {
      let hist_canvas = document.getElementById(hist_canvas_id);
      let hist_context = hist_canvas.getContext('2d');
      data.forEach(function (points) {
        let mri_x = Number(points[1]);
        let mri_y = Number(points[0]);
        let radius = 7;
        let strokeWidth = 2;
        let fill = 'red';
        let stroke = 'red';
        let color = selectColor(num_toggles);
        drawCircle(hist_context, mri_x, mri_y, radius, color, color, strokeWidth);
        mri_shapes.push({
          x: mri_x,
          y: mri_y,
          radius: radius,
          fill: color,
          stroke: color,
          strokeWith: strokeWidth
        });
        toggle_nums.push(num_toggles);
        num_toggles = num_toggles + 1;
      });
      display_hist_points = mri_shapes;
      log_toggle_nums(toggle_nums);
      log_display_points(display_hist_points, []);
      this.setState({
        hist_shapes: mri_shapes,
        mri_shapes: [],
        toggle_nums: toggle_nums
      });
    }; //////////////////////////// Functions that Trigger Python Scripts //////////////////////////


    async function predictPoints(e) {
      log_display_points(display_hist_points, []);
      let predict_btn = document.getElementById(predict_id); // predict_btn.disabled = true;

      const response = await fetch("http://localhost:5000/python", {
        method: "post",
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        // Hard Coded Values to be replaced with User Input
        // body: JSON.stringify({
        //     model_path: "C:/Users/nelsonni/OneDrive - Milwaukee School of Engineering/Documents/Research/pls_work/models/model",
        //     hist_path: hist_file,
        //     mri_path: mri_file,
        //     points_path: "C:/Users/nelsonni/OneDrive - Milwaukee School of Engineering/Documents/Research/Correct_Prostate_Points/Prostates/1102/8/predicted_histmri_points (1).csv"
        // })
        body: JSON.stringify({
          model_path: "C:/Users/nelsonni/OneDrive - Milwaukee School of Engineering/Documents/Research/pls_work/models/model",
          hist_path: hist_file,
          mri_path: mri_file,
          points_path: hist_mri_points,
          sift_points: sift_points
        })
      }); //const response = await fetch('http://localhost:5000/python');

      const body = await response.json();

      if (response.status !== 200) {
        throw Error(body.message);
      }

      console.log(body);
      let points = body.test;
      display_sift_hist_points(points);
      predict_btn.disabled = false;
    }

    async function getSiftPoints(e) {
      const response = await fetch("http://localhost:5000/sift", {
        method: "post",
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        // Hard Coded Values to be replaced with User Input
        body: JSON.stringify({
          hist_path: hist_file
        })
      }); //const response = await fetch('http://localhost:5000/python');

      const body = await response.json();

      if (response.status !== 200) {
        throw Error(body.message);
      }

      console.log(body);
      let points = body.test;
      display_predicted_mri_points(points);
      log_sift_points(points);
    }

    async function getHistFile(e) {
      const response = await fetch("http://localhost:5000/getHist", {
        method: "post",
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      }); //const response = await fetch('http://localhost:5000/python');

      const body = await response.json();

      if (response.status !== 200) {
        throw Error(body.message);
      }

      console.log(body.hist_path);
      let points = body.hist_path;
      display_predicted_mri_points(points);
    } /////////////////////// HTML Code to Construct React Components ////////////////////////////////
    // Render the application


    return /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail"
    }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h1", {
      style: {
        textAlign: "center"
      },
      className: "p-4"
    }, "Homologous Point Transformer")), /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-body"
    }, /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-nav container",
      style: {
        width: '150px'
      }
    }, hist_data && mri_data && /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-2"
    }, /*#__PURE__*/React.createElement(SiftButton, {
      onClick: getSiftPoints
    }), /*#__PURE__*/React.createElement(PredictButton, {
      onClick: predictPoints,
      id: predict_id
    })), toggle_nums.length > 0 && /*#__PURE__*/React.createElement("div", {
      className: "m-0 p -0"
    }, /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-1"
    }, /*#__PURE__*/React.createElement(ToggleGroup, {
      toggle_nums: toggle_nums,
      handlePointToggle: handlePointToggle,
      handleClickAll: handleClickALl
    })), /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-1"
    }, /*#__PURE__*/React.createElement(DownloadButton, {
      onClick: download_points,
      text: "Download Points"
    })))), /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-content"
    }, /*#__PURE__*/React.createElement("div", {
      className: "container p-0 m-0"
    }, /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-2 d-flex justify-content-center"
    }, /*#__PURE__*/React.createElement(Image, {
      id: hist_id,
      className: "col my-2 d-flex"
    }), /*#__PURE__*/React.createElement(Image, {
      id: mri_id,
      className: "col my-2 d-flex"
    })), toggle_nums.length > 0 && /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-2 d-flex justify-content-center"
    }, /*#__PURE__*/React.createElement(BackButton, {
      onClick: handlePreviousPoint
    }), /*#__PURE__*/React.createElement(ForwardButton, {
      onClick: handleNextPoint
    })), /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-2"
    }, /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: show_hist,
      promptStatement: "Choose a Histology Image:",
      acceptedFile: ".tiff"
    }), /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: show_mri,
      promptStatement: "Choose an MRI Image:",
      acceptedFile: ".nii"
    })), /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-2"
    }, /*#__PURE__*/React.createElement(FileSelect, {
      onFileSelect: display_points,
      promptStatement: "Choose Histology points:",
      acceptedFile: ".csv"
    })))), /*#__PURE__*/React.createElement("div", {
      className: "HolyGrail-right"
    }, toggle_nums.length > 0 && /*#__PURE__*/React.createElement(Switch, {
      id: mri_switch_id,
      handleClick: handleClick,
      text: "Enable Editing Points",
      className: "col"
    }))));
  }

}
class BackButton extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-primary w-25 border border-dark",
      onClick: this.props.onClick,
      style: {
        maxHWidth: '200px',
        marginTop: '10px',
        marginBottom: '15px '
      }
    }, "Previous Point");
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
class DownloadButton extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-primary w-100",
      onClick: this.props.onClick,
      style: {
        maxHWidth: '200px',
        marginTop: '10px',
        marginBottom: '15px '
      }
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
    })), this.props.text);
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
      onChange: event => this.props.onFileSelect(event.target.value),
      accept: this.props.acceptedFile
    }));
  }

}
class ForwardButton extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-primary w-25 border border-dark",
      onClick: this.props.onClick,
      style: {
        maxHWidth: '200px',
        marginTop: '10px',
        marginBottom: '15px '
      }
    }, "Next Point");
  }

}
// import Skeleton from 'react-loading-skeleton'
// import 'react-loading-skeleton/dist/skeleton.css'
class Image extends React.Component {
  render() {
    this.state = {
      margin: "auto"
    };
    return /*#__PURE__*/React.createElement("div", {
      id: this.props.id,
      className: "img-thumbnail m-4 p-0",
      style: {
        width: '500px',
        height: '500px',
        backgroundColor: "#D3D3D3",
        outlineColor: "#000000",
        outlineStyle: "solid"
      }
    });
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
class PointToggle extends React.Component {
  render() {
    const handleClick = () => {
      this.props.handlePointToggle(this.props.num);
    };

    return /*#__PURE__*/React.createElement("div", {
      className: "m-0 p-0 w-100",
      style: {
        outlineStyle: "solid",
        outlineWidth: 1
      }
    }, /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-sm btn-primary row m-0 w-100",
      onClick: handleClick,
      enabled: true
    }, "Point ", this.props.num + 1));
  }

}
class PredictButton extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("button", {
      type: "button",
      id: this.props.id,
      className: "btn btn-primary w-50 border border-dark",
      onClick: this.props.onClick,
      style: {
        maxHWidth: '200px',
        marginTop: '10px',
        marginBottom: '15px '
      }
    }, "Predict Points");
  }

}
class SiftButton extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-primary w-50 border border-dark",
      onClick: this.props.onClick,
      style: {
        maxHWidth: '200px',
        marginTop: '10px',
        marginBottom: '15px '
      }
    }, "Get Sift Points");
  }

}
class Switch extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("div", {
      className: "form-check form-switch col",
      style: {
        marginTop: '50px',
        marginBottom: '15px',
        marginLeft: 'auto',
        marginRight: '-20px'
      }
    }, /*#__PURE__*/React.createElement("input", {
      className: "form-check-input",
      type: "checkbox",
      role: "switch",
      id: this.props.id,
      onClick: this.props.handleClick
    }), /*#__PURE__*/React.createElement("label", {
      className: "form-check-label",
      htmlFor: "flexSwitchCheckDefault",
      style: {
        fontWeight: 'bold'
      }
    }, this.props.text));
  }

}
class ToggleGroup extends React.Component {
  render() {
    return /*#__PURE__*/React.createElement("div", {
      className: "row row-cols-1 w-100 mx-0 p-0",
      style: {
        overflowX: 'auto',
        maxHeight: '370px',
        outlineStyle: "solid",
        outlineWidth: 1
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "m-0 p-0 w-100",
      style: {
        outlineStyle: "solid",
        outlineWidth: 1
      }
    }, /*#__PURE__*/React.createElement("button", {
      type: "button",
      className: "btn btn-sm btn-primary row m-0 w-100",
      onClick: this.props.handleClickAll,
      enabled: true
    }, "Show All Points")), this.props.toggle_nums.map(num => /*#__PURE__*/React.createElement("div", {
      className: "col m-0 p-0 w-100"
    }, /*#__PURE__*/React.createElement(PointToggle, {
      num: num,
      handlePointToggle: this.props.handlePointToggle
    }))));
  }

}
