async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.text();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

function populateDropdown(data) {
    const lines = data.split('\n');
    const breedColumnIndex = 0; // Assuming the "Breed" column is the first column (index 0)

    const breedSelect = document.getElementById('breedSelect');
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.length > breedColumnIndex) {
            const breed = values[breedColumnIndex].trim();
            if (breed) {
                const option = document.createElement('option');
                option.value = breed;
                option.textContent = breed;
                breedSelect.appendChild(option);
            }
        }
    }
}

function createRadarChart(data) {
    const breedGroupColumnIndex = 1; // Assuming the "Breed Group" column is the third column (index 2)

    const selectedBreed = document.getElementById('breedSelect').value;
    const selectedRowIndex = data.findIndex(row => row.includes(selectedBreed));
    const breedGroup = data[selectedRowIndex].split(',')[breedGroupColumnIndex].trim();

    const radarChartContainer = document.getElementById('radarChartContainer');
    radarChartContainer.innerHTML = ''; // Clear previous chart

    // Create and configure the radar chart
    const radarChart = new Chart(radarChartContainer, {
        type: 'radar',
        data: {
            labels: ['Label 1', 'Label 2'], // Replace with your labels
            datasets: [{
                label: breedGroup,
                data: [1,2],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
            }]
        },
        options: {
            scale: {
                ticks: { beginAtZero: true },
            },
        },
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const dataURL = 'https://raw.githubusercontent.com/TinTesla/Project_4---Group_7/49e85ec0af8c10d2892e0c9c93a7a7e622d8a8db/Data/Test/Outputs/full_numberic_df.csv';
    const data = await fetchData(dataURL);
    if (data) {
        populateDropdown(data);

        const breedSelect = document.getElementById('breedSelect');
        breedSelect.addEventListener('change', () => {
            createRadarChart(data);
        });
    }
});
