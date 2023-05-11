window.addEventListener('load', async () => {
  // fetch the names.txt file
  console.log("Connected")
  const namesData = await fetch('names.txt').then(response => response.text());

  const main = document.querySelector('main');

  main.innerHTML = namesData
});