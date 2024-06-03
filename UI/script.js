const form = document.getElementById("search-form");
const documentList = document.getElementById("document-list");
const loadingIndicator = document.getElementById("loading-indicator");
const searchButton = document.getElementById("search-button");
const searchInput = document.getElementById("search-input");
const suggestionsList = document.getElementById("suggestions-list");
const suggestionsContainer = document.getElementById("suggestions-container");

// Function to render the document list
function renderDocumentList() {
  documentList.innerHTML = ""; // Clear the existing list

  const ul = document.createElement("ul");

  Object.entries(documents).forEach(([doc_id, text]) => {
    const li = document.createElement("li");
    li.textContent = `${doc_id}: ${text}`;
    ul.appendChild(li);
  });

  documentList.appendChild(ul);
}


// Function to render the suggestions list
function renderSuggestionsList() {
  suggestionsList.innerHTML = ""; // Clear the existing list
  suggestionsList.style.display = "";

  suggestions.forEach((suggestion) => {
    if (!(suggestion.trim().length === 0)) {
      const li = document.createElement("li");
      li.textContent = `${suggestion}`;
      li.addEventListener("click" , function(event){
        suggestionsList.style.display = "none";
        searchInput.value = suggestion;
      });
      suggestionsList.appendChild(li);
    }
  });
}

// Event listener for the form submission
form.addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent form submission

  suggestionsList.innerHTML = ""

  const selectedDataset = document.getElementById("datasets").value;
  const searchQuery = document.getElementById("search-input").value;

  // Disable the search button
  searchButton.disabled = true;

  // Show loading text on the search button
  searchButton.innerHTML = "Loading...";

  // Prepare the request payload
  const payload = {
    query: searchQuery,
    dataset_name: selectedDataset
  };

  // Send the POST request to the API endpoint using Axios
  axios
    .post("http://localhost:5000/query", payload)
    .then(function(response) {
      documents = response.data.relevant_docs || {};

      // Render the updated document list
      renderDocumentList();

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    })
    .catch(function(error) {
      console.log(error);

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    });
});


// Event listener for the textarea input
searchInput.addEventListener("input", function(event) {
  event.preventDefault(); // Prevent form submission

  const selectedDataset = document.getElementById("datasets").value;
  const searchQuery = document.getElementById("search-input").value;

  // Prepare the request payload
  const payload = {
    query: searchQuery,
    dataset_name: selectedDataset
  };

  // Send the POST request to the API endpoint using Axios
  axios
    .post("http://localhost:5000/suggestions", payload)
    .then(function(response) {
      suggestions = response.data.suggestions || {};
      console.log(response)

      // Render the updated suggestions list
      if (suggestions.length > 0) {
        renderSuggestionsList();
      }
    })
    .catch(function(error) {
      console.log(error);

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    });
});





// Event listener for the textarea focus
searchInput.addEventListener("focus", function(event) {
  event.preventDefault(); // Prevent form submission

  const selectedDataset = document.getElementById("datasets").value;

  // Prepare the request payload
  const payload = {
    dataset_name: selectedDataset
  };

  if(searchInput.value != "") {
    const searchQuery = document.getElementById("search-input").value;
  
    // Prepare the request payload
    const payload = {
      query: searchQuery,
      dataset_name: selectedDataset
    };

    // Send the POST request to the API endpoint using Axios
    axios
    .post("http://localhost:5000/suggestions", payload)
    .then(function(response) {
      suggestions = response.data.suggestions || {};
      console.log(response)

      // Render the updated suggestions list
      if (suggestions.length > 0) {
        renderSuggestionsList();
      }
    })
    .catch(function(error) {
      console.log(error);

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    });
  } else {

    // Send the POST request to the API endpoint using Axios
    axios
      .post("http://localhost:5000/lastSearches", payload)
      .then(function(response) {
        suggestions = response.data.suggestions || {};
        console.log(response)

        // Render the updated suggestions list
        if (suggestions.length > 0) {
          renderSuggestionsList();
        }
      })
      .catch(function(error) {
        console.log(error);

        // Enable the search button and restore its original text
        searchButton.disabled = false;
        searchButton.innerHTML = "Search";

        // Hide the loading indicator
        loadingIndicator.style.display = "none";
      });
  }
});