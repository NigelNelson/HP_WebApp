const express = require('express');
const cors = require('cors');
const spawn = require("child_process").spawn;
var bodyParser = require('body-parser')
const app = express(); //Line 2
const port = process.env.PORT || 5000; //Line 3

app.use(cors())

app.use(bodyParser.urlencoded({ extended: false }))

// parse application/json
app.use(bodyParser.json())

// This displays message that the server running and listening to specified port
app.listen(port, () => console.log(`Listening on port ${port}`)); //Line 6

// create a GET route
app.get('/express_backend', (req, res) => { //Line 9
    res.send({ express: 'TIME TO RUN INFERENCE BEOTCHHHH' }); //Line 10
});

app.post('/python', (req, res) => {
    let model_path = req.body.model_path;
    let mri_path = req.body.mri_path;
    let hist_path = req.body.hist_path;
    let points_path = req.body.points_path;

    const pythonProcess = spawn('python',["inference.py", model_path, hist_path, mri_path, points_path]);

    console.log("spawned");
    pythonProcess.stdout.on('data', (data) => {
        let points = (JSON.parse(data.toString()));
        console.log(points);
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

app.post('/sift', (req, res) => {
    let hist_path = req.body.hist_path;

    const pythonProcess = spawn('python',["sift.py", hist_path]);

    console.log("spawned");
    pythonProcess.stdout.on('data', (data) => {
        let points = (JSON.parse(data.toString()));
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