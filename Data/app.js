// Function to fetch JSON data
async function fetchData() {
  try {
      const response = await fetch("https://raw.githubusercontent.com/TinTesla/Project-3/main/Clean/json%20copies/cleaned_spotify_data.json");
      const data = await response.json();
      return data;
  } catch (error) {
      console.error("Error fetching data:", error);
  }
}

// Function to populate the drop-down menu with unique genres
function populateGenreDropdown(data) {
  const dropdown = document.getElementById("genre");
  const genres = [...new Set(data.map(entry => entry.Genre))];
  genres.forEach(genre => {
      const option = document.createElement("option");
      option.value = genre;
      option.text = genre;
      dropdown.appendChild(option);
  });
}

// Function to filter data based on selected genre and create the bubble chart
function createBubbleChart(data, selectedGenre) {
    const filteredData = selectedGenre ? data.filter(entry => entry.Genre === selectedGenre) : data;

    const bubbleData = [{
        x: filteredData.map(entry => entry.Bpm),
        y: filteredData.map(entry => entry.Valence), 
        text: filteredData.map(entry => entry.Title),
        mode: "markers",
        marker: {
            size: filteredData.map(entry => entry["Duration (in seconds)"]/10),
            color: filteredData.map(entry => getYearColor(entry.Year)),
            colorscale: "Rainbow"
        }
    }];

    const bubbleLayout = {
        xaxis: { title: "Bpm" },
        yaxis: { title: "Valence" }
    };

    Plotly.newPlot("bubble", bubbleData, bubbleLayout);
}

// Function to get color based on year
function getYearColor(year) {
  if (year <= 1999) {
      return "red";
  } else if (2000 <= year && year <= 2004) {
      return "orange";
  } else if (2005 <= year && year <= 2009) {
      return "yellow";
  } else if (2010 <= year && year <= 2014) {
      return "green";
  } else if (2015 <= year && year <= 2019) {
      return "blue";
  } else if (year >= 2020) {
      return "purple";
  } else {
      return "black";
  }
}

// Function to update the graph when the genre is changed
function updateGraph() {
  const selectedGenre = document.getElementById("genre").value;
  createBubbleChart(data, selectedGenre);
}

// Fetch data and initialize the graph
let data;
fetchData()
  .then(result => {
      data = result;
      populateGenreDropdown(data);
      createBubbleChart(data, null);
  })
  .catch(error => console.error("Error:", error));
