// Class: SE2840 - Person Tracker
// Name: Nigel Nelson
// Class Section: 021

/**
 * Class that encapsulates the variables and functions that pertain to the person tracking
 * functionality.
 */
class PersonTracker {
    constructor() {
        this.map = this.initMap();
        this.markerLayer = L.layerGroup().addTo(this.map);
        this.filter = "Show All";
    }

    /**
     * Initializes the leaflet map
     * @return {*} the leaflet map instance
     */
    initMap() {
        const usa_geographic_center = [39.8283, -98.5795]
        const map = L.map('map').setView(usa_geographic_center, 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        return map;
    }

    /**
     * Assigns the corresponding event handlers to the input elements included in the Person Tracker App
     */
    initInputElements() {

        const radios = document.getElementsByName("inlineRadioOptions");
        radios.forEach((radio) => {
            radio.onclick = () => this.filterSelection(radio);
        });

        const displayBtn = document.getElementById("displayButton");
        displayBtn.onclick = () => this.onDisplay();

        const sortBtn = document.getElementById("sortButton");
        sortBtn.onclick = () => this.onSort();
    }

    /**
     * Responds to the sort button being clicked. Responsible for getting the sorting category and the sorting
     * direction, and passing this information to the sortTable method
     */
    onSort(){
        const sortCatOptions = document.getElementById("sortType");
        const sortCat = sortCatOptions.options[sortCatOptions.selectedIndex].value;

        let sortedCol = 0;

        if(sortCat === "name"){
            sortedCol = 0;
        }else if (sortCat === "age"){
            sortedCol = 1;
        } else{
            // Remaining option is sorting by hometown
            sortedCol = 2;
        }

        const sortDirOptions = document.getElementById("sortDirection");
        const sortDir = sortDirOptions.options[sortDirOptions.selectedIndex].value;

        const isAscendingSort = sortDir === "inc";

        this.sortTable(sortedCol, isAscendingSort);
    }

    /**
     * Responsible for sorting the HTML table by comparing the values of the specified column, and switching row
     * indices when two sequential rows are out of order.
     * Resource used: https://www.w3schools.com/howto/howto_js_sort_table.asp
     *
     * @param sortedCol The index of the column that contains the data that the user would like to sort by
     * @param isAscendingSort If the desired sort direction is ascending
     */
    sortTable(sortedCol, isAscendingSort) {
        let table = document.getElementById("table");
        let switching = true;

        // Loop until no switching has to be performed
        while (switching) {
            let i, shouldSwitch;

            // Assume no switches need to take place
            switching = false;
            let rows = table.rows;

            for (i = 0; i < rows.length - 1; i++) {

                // Assume the rows shouldn't be switched
                shouldSwitch = false;

                // Get two sequential rows
                let val1 = rows[i].getElementsByTagName("TD")[sortedCol];
                let val2 = rows[i + 1].getElementsByTagName("TD")[sortedCol];

                // Check which direction to sort
                if(isAscendingSort){
                    // If the first rows value is greater than the next's, mark that it needs to be switched
                    if (val1.innerHTML.toLowerCase() > val2.innerHTML.toLowerCase()) {
                        // If so, mark as a switch and break the loop:
                        shouldSwitch = true;
                        break;
                    }
                } else{
                    if (val1.innerHTML.toLowerCase() < val2.innerHTML.toLowerCase()) {
                        // If so, mark as a switch and break the loop:
                        shouldSwitch = true;
                        break;
                    }
                }
            }
            if (shouldSwitch) {
                // Switch the two rows
                rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                switching = true;
            }
        }
    }

    /**
     * Method invoked when a radio button is clicked. Responsible for setting the PersonTracker's desired filter
     * value and toggles the disabled state of the text input accordingly
     * @param radioBtn The radio button clicked
     */
    filterSelection(radioBtn) {
        const idVal = radioBtn.id;

        // Get the label corresponding to the clicked radion button
        const query = "label[for='" + idVal + "']"
        this.filter = document.querySelector(query).innerHTML;

        const textFilter = document.getElementById("filterText");

        // If the "Show All" option is selected, clear and disable the input text
        if(idVal === "allRadio"){
            textFilter.value = "";
            textFilter.disabled = true;
        } else{
            textFilter.disabled = false;
        }
    }

    /**
     * Method invoked when the "Display" button is clicked. Responsible for getting the radio button selected,
     * and the filter text's value, and passing that information to the makeServerRequest() method
     */
    onDisplay() {
        const filterValue = document.getElementById("filterText").value;
        let path = "";
        if (this.filter === "Show All") {
            path = "all";
        } else if (this.filter === "First Name") {
            path = "firstname";
        } else if (this.filter === "Last Name") {
            path = "lastname";
        } else if (this.filter === "Age") {
            path = "age";
        } else {
            path = "hometown";
        }
        this.makeSeverRequest(path, filterValue);
    }

    /**
     * Responsible for requesting information from the server that reflects the user's desired display parameters
     * @param path The URL path corresponding to the radio button selected
     * @param filterValue The contents of the filter value text
     */
    makeSeverRequest(path, filterValue) {

        const query = "filtervalue=" + filterValue;

        $.ajax({
            type: "GET", // Specify to use a GET request
            url: `http://localhost:3000/${path}`, // The resource being requested
            data: query, // The query values that reflect the filter value entered by the user
            async: true,        // Make the request async
            dataType: "json",   // Get results via JSON
            success: (response, textStatus, jqXHR) => {this.readResponse(response, textStatus, jqXHR)},
            error: (response, textStatus, jqXHR) => {this.handleError(response, textStatus, jqXHR)},
        });
    }

    /**
     * Method invoked when the request was successful. Responsible for parsing the response JSON, and either displaying
     * the data that was returned, or displaying an error if the response indicates that there was an error in the
     * request
     * @param response the data returned by the request
     * @param textStatus string status of the response
     * @param jqXHR XMLHttpRequest object
     */
    readResponse(response, textStatus, jqXHR) {

        // Clear data from previous queries
        const table = document.getElementById("table");
        table.innerHTML = "";
        this.markerLayer.clearLayers();
        this.removeError();

        // If the status is error, pass the params to the handleError() method
        if (response.status === "error"){
            this.handleError(response, textStatus, jqXHR);
            return;
        }

        // Iterate for all the People returned in the response
        for (let i = 0; i < response.length; i++) {

            // Get the attributes associated with the Person JSON object
            const fullName = response.values[i].firstname + " " + response.values[i].lastname;
            const lat = response.values[i].location.lat;
            const lon = response.values[i].location.lon;
            const hometown = response.values[i].hometown;

            // Insert the Person into the HTML table
            const row = table.insertRow(-1);
            const nameCol = row.insertCell(-1);
            nameCol.innerText = fullName;
            const ageCol = row.insertCell(-1);
            ageCol.innerText = response.values[i].age;
            const hometownCol = row.insertCell(-1);
            hometownCol.innerText = response.values[i].hometown;
            this.addMarker(lat, lon, fullName, hometown);

            // Pan to the Person location on the map when the row is clicked
            row.onclick = (() => {
                this.map.panTo([lat, lon])
            });
        }

        // If the response contained no People, hide the sorting interface
        if(response.values.length !== 0){
            this.changeSortingVisibility(false);
        } else{
            this.changeSortingVisibility(true);
        }
    }

    /**
     * Adds a marker on the map for a Person with their name displayed, and a popup of their hometown when the marker
     * is clicked
     * @param lat the latitude of the Person
     * @param lon the longitude of the Person
     * @param fullName the full name of the Person
     * @param hometown the hometown of the Person
     */
    addMarker(lat, lon, fullName, hometown) {
        const marker = L.marker([lat, lon]).addTo(this.markerLayer);
        marker.bindPopup("Hometown: " + hometown);
        marker.bindTooltip(fullName, {
            offset: L.point(-15, 25),
            direction: 'bottom',
            permanent: true
        }).openTooltip();
    }

    /**
     * Changes the sorting interface's visibility depending on whether the request resulted in any people being
     * returned
     * @param isResponseEmpty is the response void of any people
     */
    changeSortingVisibility(isResponseEmpty){
        const sortDiv = document.getElementById("sortControl");
        if(isResponseEmpty){
            sortDiv.classList.add("visually-hidden");
        } else{
            sortDiv.classList.remove("visually-hidden");
        }
    }

    /**
     * Responsible for removing any error that is currently displayed
     */
    removeError() {
        const errorDisplay = document.getElementById("messages");
        errorDisplay.innerText = "";
        errorDisplay.classList.add("visually-hidden");
    }

    /**
     * Responsible for handling errors that resulted from faulty requests
     * @param response the data returned by the request
     * @param textStatus string status of the response
     * @param jqXHR XMLHttpRequest object
     */
    handleError(response, textStatus, jqXHR) {
        const table = document.getElementById("table");
        table.innerHTML = "";
        this.markerLayer.clearLayers();
        this.changeSortingVisibility(true);

        const errorDisplay = document.getElementById("messages");
        errorDisplay.innerText = "Error: " + response.message;
        errorDisplay.classList.remove("visually-hidden");
    }
}

/**
 * Upon loading the window, a new PersonTracker is created and the input elements are initialized.
 */
window.onload = () => {
    const personTracker = new PersonTracker();
    personTracker.initInputElements()
}
