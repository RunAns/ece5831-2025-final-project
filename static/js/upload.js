(() => {
  const input = document.getElementById("myfile");
  const label = document.getElementById("fileLabel");
  const btn = document.getElementById("predictBtn");
  const audio = document.getElementById("previewAudio");
  const hint = document.getElementById("previewHint");

  if (!input) return;

  input.addEventListener("change", () => {
    const f = input.files && input.files[0];
    if (!f) {
      label.textContent = "No file selected";
      btn.disabled = true;
      if (audio) audio.style.display = "none";
      if (hint) hint.style.display = "block";
      return;
    }

    label.textContent = f.name;

    const ok = /\.(wav|mp3)$/i.test(f.name);
    btn.disabled = !ok;

    if (!ok) {
      if (hint) hint.textContent = "Please select a .wav or .mp3 file.";
      if (hint) hint.style.display = "block";
      if (audio) audio.style.display = "none";
      return;
    }

    // preview audio
    if (audio) {
      audio.src = URL.createObjectURL(f);
      audio.style.display = "block";
    }
    if (hint) hint.style.display = "none";
  });
})();
