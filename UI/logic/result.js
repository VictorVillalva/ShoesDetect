document.getElementById("regreso").addEventListener("click", function(){
    window.location.href = "detector.html";
})

var fotoDataUrl = localStorage.getItem('foto');
if (fotoDataUrl) {
    document.getElementById('imagenSeleccionada').src = fotoDataUrl;
} else {
    alert('La imagen no se encontró.');
}