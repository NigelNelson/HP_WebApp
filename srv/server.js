const express = require('express');
const cors = require('cors');
const spawn = require("child_process").spawn;
var bodyParser = require('body-parser')
const app = express(); //Line 2
const port = process.env.PORT || 5000; //Line 3

// let sift_points = [];

app.use(cors())

app.use(bodyParser.urlencoded({ extended: false }))
app.use(bodyParser({limit: '50mb'}));
// parse application/json
app.use(bodyParser.json())

// This displays message that the server running and listening to specified port
app.listen(port, () => console.log(`Listening on port ${port}`)); //Line 6


app.post('/python', (req, res) => {
    let model_path = req.body.model_path;
    let mri_path = req.body.mri_path;
    let hist_path = req.body.hist_path;
    let points_path = req.body.points_path;
    let sift_points = req.body.sift_points;

    console.log(sift_points);

    const process = spawn('python',["inference.py", model_path]);

    process.stdin.write(hist_path + '\n' + mri_path + '\n' + points_path + '\n' + sift_points);
    process.stdin.end();

    console.log("spawned");
    process.stdout.on('data', (data) => {
        let points = (JSON.parse(data.toString()));
        console.log(points);
        res.send({'test': points});
        console.log("sent");
    });//Line 10

    process.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    process.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
}); //Line

app.post('/sift', (req, res) => {
    let hist_path = req.body.hist_path;

    const pythonProcess = spawn('python',["sift.py"]);

    pythonProcess.stdin.write(hist_path);
    pythonProcess.stdin.end();

    console.log("spawned");
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
    });
}); //Line


app.post('/getHist', (req, res) => {
    let hist_path = req.body.hist_path;

    const pythonProcess = spawn('python',["getHist.py"]);

    console.log("spawned");
    pythonProcess.stdout.on('data', (data) => {
        let path = (JSON.parse(data.toString()));
        res.send({'hist_path': path});
        console.log("sent");
    });//Line 10

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
}); //Line