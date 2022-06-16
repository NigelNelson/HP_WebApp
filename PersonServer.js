// Class: SE2840 - Person Tracker
// Name: Nigel Nelson
// Class Section: 021

// Import the express library
const express = require('express');

const personData = require('./PersonData.js');
const webpagedir = `${__dirname}/public`;

// Create a new express application
const app = express();

// Instruct the app to listen on port 3000
app.listen(3000, () => {
    console.log("Server is running at http://localhost:3000");
});

// Set a static route for root (/) to send PersonTracker.html
app.get("/", (request, response) => {
    response.sendFile(`${webpagedir}/index.html`);
});

// Set a static route for all resources that don't have an explicit route
// to use the static directory webpagedir
app.use(express.static(webpagedir));

// Response for "Show All" option, which returns all People contained in personData
app.get("/all", (request, response) => {
    response.json({status: "success", length: personData.length, "values": personData});
});

// Response for when People are filtered by first name
app.get("/getImg", (request, response) => {
    try{
        path = request.query.path
        console.log(path)
        // response.json({"path": path})
        response.sendFile(path)

    } catch (error){
        response.json({
            status: "error",
            message: error.message
        });
    }
});

// Response for when People are filtered by last name
app.get("/lastname", (request, response) => {
    // try{
    //     let lastName = request.query.filtervalue.toLowerCase();
    //     const results = []
    //
    //     if(!lastName){
    //         throw Error("Last Name cannot be an empty string");
    //     }
    //
    //     personData.forEach(value => {
    //         // Add a Person to the results if their last name includes the filter value
    //         if(value.lastname.toLowerCase().includes(lastName)){
    //             results.push(value);
    //         }
    //     });
    //
    //     response.json({status: "success", length: results.length, "values": results});
    //
    // } catch (error){
    //     response.json({
    //         status: "error",
    //         message: error.message
    //     });
    // }
    try{
        path = request.query.path
        response.sendFile(path)

    } catch (error){
        response.json({
            status: "error",
            message: error.message
        });
    }
});

// Response for when People are filtered by age
app.get("/age", (request, response) => {
    try{
        let value = request.query.filtervalue;
        const results = []

        // Check that age is not empty
        if(!value){
            throw Error("Age cannot be empty");
        }

        const age = parseInt(value);

        // Check that age is a number and greater than 0
        if(isNaN(age) || age < 0){
            throw Error("Age must be a positive integer")
        }

        // Check that age is an integer
        if(!Number.isInteger(parseFloat(value))) {
            throw Error("Age cannot be a floating point number");
        }

        // Add all people who's age is an exact match
        personData.forEach(value => {
            if(value.age === age){
                results.push(value);
            }
        });

        response.json({status: "success", length: results.length, "values": results});

    } catch (error){
        response.json({
            status: "error",
            message: error.message
        });
    }
});

// Response for when People are filtered by hometown
app.get("/hometown", (request, response) => {
    try{
        let hometown = request.query.filtervalue.toLowerCase();
        const results = []

        if(!hometown){
            throw Error("Last Name cannot be an empty string");
        }

        // Add a Person to the results if their hometown includes the filter value
        personData.forEach(value => {
            if(value.hometown.toLowerCase().includes(hometown)){
                results.push(value);
            }
        });

        response.json({status: "success", length: results.length, "values": results});

    } catch (error){
        response.json({
            status: "error",
            message: error.message
        });
    }
});

// Respond with a 404 not found error if a requested file does not exist
app.use(function(req, res, next) {
    res.status(404).send("HTTP 404 Not Found");
});