const express = require('express');
const cors = require('cors');
const spawn = require("child_process").spawn;
var bodyParser = require('body-parser')
const app = express(); //Line 2
const port = process.env.PORT || 5000; //Line 3

const model_path = "C:/Users/nelsonni/OneDrive - Milwaukee School of Engineering/Documents/Research/HPT_App/models/model";

// let sift_points = [];

app.use(cors())

app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser({limit: '50mb'}));
// parse application/json
app.use(bodyParser.json())

// This displays message that the server running and listening to specified port
app.listen(port, () => console.log(`Listening on port ${port}`)); //Line 6

// Start the inference script so it's ready for any requests
let inferenceProcess = spawn('python',["inference.py", model_path]);
let inferenceRunning = true;
console.log("spawned inference process");
// Start the SIFT script so it's ready for any requests
let pythonProcess = spawn('python',["sift.py"]);
let siftRunning = true;
console.log("spawned sift process");


app.post('/python', (req, res) => {
    /**
     * Endpoint to handle inference requests
     */
    if(!inferenceRunning){
        inferenceProcess = spawn('python',["inference.py", model_path]);
        console.log("spawned inference process");
    }
    console.log("Running inference process");
    let mri_path = req.body.mri_path;
    let hist_path = req.body.hist_path;
    let points_path = req.body.points_path;
    let sift_points = req.body.sift_points;

    console.log(sift_points);

    // send the collected paths to the inference process
    inferenceProcess.stdin.write(hist_path + '\n' + mri_path + '\n' + points_path + '\n' + sift_points);
    inferenceProcess.stdin.end();

    // Send the response points
    inferenceProcess.stdout.on('data', (data) => {
        let points = (JSON.parse(data.toString()));
        console.log(points);
        res.send({'test': points});
        console.log("sent");
    });//Line 10

    inferenceProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    inferenceProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        inferenceRunning = false;
    });
}); //Line

app.post('/sift', (req, res) => {
    /**
     * Endpoint to handle SIFT requests
     */
    if(!siftRunning){
        pythonProcess = spawn('python',["sift.py"]);
        console.log("spawned sift process");
    }
    console.log("Running sift process");
    let hist_path = req.body.hist_path;

    // Send the path to the histology image to the SIFT script
    pythonProcess.stdin.write(hist_path);
    pythonProcess.stdin.end();

    // Respond with the SIFT points
    pythonProcess.stdout.on('data', (data) => {
        points = (JSON.parse(data.toString()));
        res.send({'test': points});
        console.log("sent");
    });//Line 10

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        siftRunning = false;
    });
}); //Line