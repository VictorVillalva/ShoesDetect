document.getElementById('img').addEventListener('change', function(event) {
    var fotoInput = event.target.files[0];
    if (fotoInput) {
        var reader = new FileReader();
        reader.onload = function(event) {
            localStorage.setItem('foto', event.target.result);
            window.location.href = 'result.html'
        };
        reader.readAsDataURL(fotoInput);
    } else {
        alert('Por favor, seleccione una foto.');
    }
});
