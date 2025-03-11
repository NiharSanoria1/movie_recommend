const movieInput = document.getElementById('movieInput');
const recommendButton = document.getElementById('recommendButton');
const recommendationsDiv = document.getElementById('recommendations');
const recommendationList = document.getElementById('recommendationList');

recommendButton.addEventListener('click', () => {
    const movieName = movieInput.value;
    if (movieName.trim() === "") {
        alert("Please enter a movie name.");
        return;
    }
    fetch(`/recommend?movie=${encodeURIComponent(movieName)}`)
        .then(response => {
            if (!response.ok) {
                if (response.status === 404){
                  return response.json().then(data =>{
                    throw new Error(data.error);
                  });
                }
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            recommendationList.innerHTML = ''; // Clear previous results
            if (data.recommendations.length === 0) {
                const li = document.createElement('li');
                li.textContent = `No recommendations found for ${movieName}`;
                recommendationList.appendChild(li);
            } else {
              data.recommendations.forEach(movie => {
                  const li = document.createElement('li');
                  li.textContent = movie;
                  recommendationList.appendChild(li);
              });
            }
            recommendationsDiv.style.display = 'block';
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
            recommendationList.innerHTML = `<li>${error}</li>`;
            recommendationsDiv.style.display = 'block';
        });
});
