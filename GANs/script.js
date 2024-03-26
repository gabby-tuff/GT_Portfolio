const button = document.getElementById("gen");
const resultDiv = document.getElementById('result');

button.addEventListener('click', async () => {
  const dataToSend = {  // Example data to send
    message: ""
  };

  try {
    const response = await fetch('/app', {
      method: 'POST',
      url: 'http://127.0.0.1:5500/app',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dataToSend)
    });

    const processedData = await response.json();
    resultDiv.textContent = processedData.message;  // Update UI
  } catch (error) {
    console.error('Error:', error);
  }
});