// // Define a function to fetch symptoms data from the server
// function fetchSymptoms() {
//     fetch('/symptoms')
//     .then(async response => {
//         const res=await  response.json()
//         const select = document.getElementById('symptoms');
//         res.forEach(symptom => {
//             const option = document.createElement('option');
//             option.text = symptom;
//             select.add(option);
//         });
//     } )
//     // .then(data => {
//     //     const select = document.getElementById('symptoms');
//     //     data.forEach(symptom => {
//     //         const option = document.createElement('option');
//     //         option.text = symptom;
//     //         console.log(option)
//     //         select.add(option);
//     //     });
//     // })
//     .catch(error => console.error('Error fetching symptoms:', error));
// }



// Fetch symptoms from the server
let allSymptoms = [];

function fetchSymptoms() {
    fetch('/symptoms')
        .then(async response => {
            allSymptoms = await response.json();
        })
        .catch(error => console.error('Error fetching symptoms:', error));
}

// Filter symptoms based on user input
function filterSymptoms() {
    const input = document.getElementById('symptom-search').value.toLowerCase();
    const dropdown = document.getElementById('symptom-dropdown');
    dropdown.innerHTML = '';

    const filteredSymptoms = allSymptoms.filter(symptom => symptom.toLowerCase().includes(input));

    filteredSymptoms.forEach(symptom => {
        const option = document.createElement('div');
        option.textContent = symptom;
        option.classList.add('dropdown-item');
        option.onclick = () => addSymptom(symptom);
        dropdown.appendChild(option);
    });
}

// Add symptom to selected list
function addSymptom(symptom) {
    const selectedContainer = document.getElementById('selected-symptoms');
    const symptomTag = document.createElement('div');
    symptomTag.classList.add('symptom-tag');
    symptomTag.textContent = symptom;

    const removeBtn = document.createElement('span');
    removeBtn.textContent = 'x';
    removeBtn.classList.add('remove-btn');
    removeBtn.onclick = () => selectedContainer.removeChild(symptomTag);

    symptomTag.appendChild(removeBtn);
    selectedContainer.appendChild(symptomTag);

    // Clear the search input and dropdown
    document.getElementById('symptom-search').value = '';
    document.getElementById('symptom-dropdown').innerHTML = '';
}



// Call the fetchSymptoms function when the DOM content is loaded
document.getElementById('symptom-search').addEventListener('input', filterSymptoms);
document.addEventListener('DOMContentLoaded', fetchSymptoms);

// Define a function to make a prediction request to the server
function predictDisease() {
    // const selectedSymptoms = Array.from(document.querySelectorAll('#symptoms option:checked')).map(option => option.value);
    const selectedSymptoms = Array.from(document.querySelectorAll('#selected-symptoms .symptom-tag'))
    .map(tag => tag.firstChild.textContent);

    console.log(JSON.stringify(selectedSymptoms));
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(selectedSymptoms)
    })
    .then(response => response.json())
    .then(data => {
        const resultsContainer = document.getElementById('prediction-results');
        resultsContainer.innerHTML = '';

        data.forEach(prediction => {
            console.log(prediction)
            const predictionContainer = document.createElement('div');
    predictionContainer.style.border = '1px solid black';  // Add border styling
    predictionContainer.style.padding = '10px';            // Optional: Add padding for better spacing
    predictionContainer.style.marginBottom = '10px';       // Optional: Add margin for spacing between predictions

            const diseaseName = document.createElement('h3');
            diseaseName.textContent = prediction.disease;
            resultsContainer.appendChild(diseaseName);
            const probability = document.createElement('p');
            probability.textContent = 'Probability: ' + (prediction.probability * 100).toFixed(2) + '%';
            resultsContainer.appendChild(probability);

            const description = document.createElement('p');
            description.textContent = 'Description: ' + prediction.description;
            resultsContainer.appendChild(description);

            if (prediction.precautions.length > 0) {
                const precautionsHeader = document.createElement('p');
                precautionsHeader.textContent = 'Precautions:';
                resultsContainer.appendChild(precautionsHeader);

                const precautionsList = document.createElement('ul');
                prediction.precautions.forEach(precaution => {
                    const precautionItem = document.createElement('li');
                    precautionItem.textContent = precaution;
                    precautionsList.appendChild(precautionItem);
                });
                resultsContainer.appendChild(precautionsList);
            }

    // Append the container to the results container
    resultsContainer.appendChild(predictionContainer);
        });
    })
    .catch(error => console.error('Error predicting disease:', error));
    // console.log("ghgg");
}



// const predictDiseases=()=>{
//     alert("found")
// }
// Fetch symptoms data when the page loads
window.onload = fetchSymptoms;
