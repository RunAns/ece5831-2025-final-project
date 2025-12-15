(() => {
  const input = document.getElementById("audioFile");
  const help = document.getElementById("fileHelp");
  const btn = document.getElementById("predictBtn");
  const audio = document.getElementById("audioPreview");
  const hint = document.getElementById("audioHint");

  if (!input) return;

  input.addEventListener("change", () => {
    const f = input.files && input.files[0];
    if (!f) {
      help.textContent = "No file selected.";
      btn.disabled = true;
      audio.classList.add("d-none");
      hint.classList.remove("d-none");
      return;
    }

    help.textContent = f.name;

    const ok = /\.(wav|mp3)$/i.test(f.name);
    btn.disabled = !ok;

    if (!ok) {
      hint.textContent = "Please select a .wav or .mp3 file.";
      audio.classList.add("d-none");
      hint.classList.remove("d-none");
      return;
    }

    const url = URL.createObjectURL(f);
    audio.src = url;
    audio.classList.remove("d-none");
    hint.classList.add("d-none");
  });
})();
