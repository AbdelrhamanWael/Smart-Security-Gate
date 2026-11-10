/**
 * app.js - Biometric Security Gate & Google-Sheets-style RBAC Dashboard
 * Supports:
 *   - Real-time Webcam streaming, Snapshot Capture & File Upload for Face.
 *   - Real-time Microphone recording & Audio File Upload for Voice.
 *   - Fingerprint image upload & Curated dataset sample selection.
 *   - Individual modality verification tests (Mode A, Mode B, Fingerprint).
 *   - Cascaded pipeline verification (Mode D).
 *   - Role-Based Access Control (Super Admin vs. Operator).
 *   - Interactive Server Inventory Data Sheet with search, filter, and inline edit.
 */

// State
let claimedUser = "member1";
let currentSessionToken = null;
let currentRoleKey = "guest";
let presetsData = null;
let datasetFilesData = null;

// Modality Input Modes
let faceInputMode = "live";       // 'live', 'upload', 'dataset'
let voiceInputMode = "live";     // 'live', 'upload', 'dataset'
let fpInputMode = "upload";       // 'upload', 'dataset'

// Live / Captured Buffers & Uploaded Files
let webcamStream = null;
let capturedFaceBlob = null;
let uploadedFaceFile = null;
let selectedFacePath = null;

let recordedAudioBlob = null;
let uploadedVoiceFile = null;
let selectedVoicePath = null;

let uploadedFpFile = null;
let selectedFpPath = null;

// Audio context & recording
let audioRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyserNode = null;
let waveAnimFrame = null;

let allInventoryRows = [];

// Initialize on Load
document.addEventListener("DOMContentLoaded", async () => {
  startClock();
  await loadPresets();
  await loadDatasetFiles();
  setupAudioVisualizer();
  await fetchDashboardData();
  
  // Set default dataset paths initially
  initDefaultDropdowns();
});

function startClock() {
  const clockEl = document.getElementById("clockDisplay");
  setInterval(() => {
    const d = new Date();
    clockEl.innerText = d.toTimeString().split(" ")[0];
  }, 1000);
}

// ---------------------------------------------------------------------------
// Load API Presets and Dataset Files
// ---------------------------------------------------------------------------

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    presetsData = await res.json();
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

async function loadDatasetFiles() {
  try {
    const res = await fetch("/api/dataset/files");
    datasetFilesData = await res.json();
    populateDatasetDropdowns();
  } catch (err) {
    console.error("Failed to load dataset files:", err);
  }
}

function populateDatasetDropdowns() {
  if (!datasetFilesData) return;

  // Face dropdown
  const faceSelect = document.getElementById("faceSelectDropdown");
  faceSelect.innerHTML = '<option value="">-- اختر صورة وجه من الداتا --</option>';
  datasetFilesData.faces.forEach(item => {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.innerText = `${item.label} (${item.path})`;
    faceSelect.appendChild(opt);
  });

  // Voice dropdown
  const voiceSelect = document.getElementById("voiceSelectDropdown");
  voiceSelect.innerHTML = '<option value="">-- اختر ملف صوتي من الداتا --</option>';
  datasetFilesData.voices.forEach(item => {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.innerText = `${item.label} (${item.path})`;
    voiceSelect.appendChild(opt);
  });

  // Fingerprint dropdown
  const fpSelect = document.getElementById("fpSelectDropdown");
  fpSelect.innerHTML = '<option value="">-- اختر صورة بصمة من الداتا --</option>';
  datasetFilesData.fingerprints.forEach(item => {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.innerText = `${item.label} (${item.path})`;
    fpSelect.appendChild(opt);
  });
}

function initDefaultDropdowns() {
  if (!presetsData) return;
  const userPresets = presetsData[claimedUser] || presetsData["member1"];
  selectedFacePath = userPresets.face_sample;
  selectedVoicePath = userPresets.voice_sample;
  selectedFpPath = userPresets.fp_sample;

  // Set dropdown values
  const faceSelect = document.getElementById("faceSelectDropdown");
  if (faceSelect) faceSelect.value = selectedFacePath;
  const voiceSelect = document.getElementById("voiceSelectDropdown");
  if (voiceSelect) voiceSelect.value = selectedVoicePath;
  const fpSelect = document.getElementById("fpSelectDropdown");
  if (fpSelect) fpSelect.value = selectedFpPath;
}

function selectClaimedUser(userId, btnEl) {
  claimedUser = userId;
  document.querySelectorAll(".user-pill").forEach(p => p.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");

  initDefaultDropdowns();
  resetVerificationState();
}

function resetVerificationState() {
  document.querySelectorAll(".stage-card").forEach(c => {
    c.classList.remove("passed", "failed");
  });
  document.getElementById("badgeFace").className = "stage-badge";
  document.getElementById("badgeFace").innerText = "STANDBY";
  document.getElementById("badgeVoice").className = "stage-badge";
  document.getElementById("badgeVoice").innerText = "STANDBY";
  document.getElementById("badgeFp").className = "stage-badge";
  document.getElementById("badgeFp").innerText = "STANDBY";

  document.getElementById("faceBarFill").style.width = "0%";
  document.getElementById("voiceBarFill").style.width = "0%";
  document.getElementById("fpBarFill").style.width = "0%";
  document.getElementById("faceScoreVal").innerText = "0.0%";
  document.getElementById("voiceScoreVal").innerText = "0.0%";
  document.getElementById("fpScoreVal").innerText = "0.0%";

  document.getElementById("timeFace").innerText = "--";
  document.getElementById("exitFace").innerText = "--";
  document.getElementById("timeVoice").innerText = "--";
  document.getElementById("exitVoice").innerText = "--";
  document.getElementById("timeFp").innerText = "--";
  document.getElementById("exitFp").innerText = "--";
  document.getElementById("timeTotal").innerText = "--";

  const banner = document.getElementById("decisionBanner");
  banner.className = "decision-banner idle";
  document.getElementById("decisionTitle").innerText = "GATE SECURED";
  document.getElementById("decisionDesc").innerText = "Awaiting biometric inputs for identity evaluation.";
  document.getElementById("dashboardUnlockBox").style.display = "none";
}

// ---------------------------------------------------------------------------
// Modality Input Mode Switchers
// ---------------------------------------------------------------------------

function setFaceInputMode(mode) {
  faceInputMode = mode;
  document.getElementById("faceTabLive").classList.toggle("active", mode === "live");
  document.getElementById("faceTabUpload").classList.toggle("active", mode === "upload");
  document.getElementById("faceTabDataset").classList.toggle("active", mode === "dataset");

  document.getElementById("faceUploadSection").style.display = mode === "upload" ? "flex" : "none";
  document.getElementById("faceDatasetSection").style.display = mode === "dataset" ? "flex" : "none";

  const video = document.getElementById("webcamVideo");
  const imgFace = document.getElementById("presetFaceImg");
  const placeholder = document.getElementById("camPlaceholder");
  const canvas = document.getElementById("capturedFaceCanvas");
  const btnStart = document.getElementById("btnStartCamera");
  const btnCap = document.getElementById("btnCaptureFace");

  if (mode === "live") {
    btnStart.style.display = "inline-flex";
    if (webcamStream) {
      video.style.display = "block";
      placeholder.style.display = "none";
      btnCap.style.display = "inline-flex";
    } else {
      placeholder.style.display = "flex";
      video.style.display = "none";
      btnCap.style.display = "none";
    }
    imgFace.style.display = "none";
    canvas.style.display = "none";
  } else if (mode === "upload") {
    stopCamera();
    btnStart.style.display = "none";
    btnCap.style.display = "none";
    if (uploadedFaceFile) {
      imgFace.style.display = "block";
      placeholder.style.display = "none";
    } else {
      imgFace.style.display = "none";
      placeholder.style.display = "flex";
      placeholder.querySelector("div:nth-child(2)").innerText = "يرجى رفع صورة الوجه من جهازك";
      placeholder.querySelector("div:nth-child(3)").innerText = "اضغط على زر 'اختر ملف صورة'";
    }
  } else if (mode === "dataset") {
    stopCamera();
    btnStart.style.display = "none";
    btnCap.style.display = "none";
    if (selectedFacePath) {
      imgFace.src = "/" + selectedFacePath;
      imgFace.style.display = "block";
      placeholder.style.display = "none";
    }
  }
}

function setVoiceInputMode(mode) {
  voiceInputMode = mode;
  document.getElementById("voiceTabLive").classList.toggle("active", mode === "live");
  document.getElementById("voiceTabUpload").classList.toggle("active", mode === "upload");
  document.getElementById("voiceTabDataset").classList.toggle("active", mode === "dataset");

  document.getElementById("voiceUploadSection").style.display = mode === "upload" ? "flex" : "none";
  document.getElementById("voiceDatasetSection").style.display = mode === "dataset" ? "flex" : "none";
  document.getElementById("btnRecordVoice").style.display = mode === "live" ? "inline-flex" : "none";

  const statusTxt = document.getElementById("voiceStatusText");
  if (mode === "live") {
    statusTxt.innerText = "جاهز: انطق عبارة 'Open security gate now'";
  } else if (mode === "upload") {
    statusTxt.innerText = uploadedVoiceFile ? `ملف مرفوع: ${uploadedVoiceFile.name}` : "يرجى اختيار ملف صوتي من جهازك (WAV)";
  } else if (mode === "dataset") {
    statusTxt.innerText = selectedVoicePath ? `عينة محددة: ${selectedVoicePath.split("/").pop()}` : "اختر عينة صوتية من القائمة";
  }
}

function setFpInputMode(mode) {
  fpInputMode = mode;
  document.getElementById("fpTabUpload").classList.toggle("active", mode === "upload");
  document.getElementById("fpTabDataset").classList.toggle("active", mode === "dataset");

  document.getElementById("fpUploadSection").style.display = mode === "upload" ? "flex" : "none";
  document.getElementById("fpDatasetSection").style.display = mode === "dataset" ? "flex" : "none";

  const preview = document.getElementById("fpPreviewImg");
  const placeholder = document.getElementById("fpPlaceholder");

  if (mode === "upload") {
    if (uploadedFpFile) {
      preview.style.display = "block";
      placeholder.style.display = "none";
    } else {
      preview.style.display = "none";
      placeholder.style.display = "flex";
      placeholder.querySelector("div:last-child").innerText = "يرجى رفع صورة البصمة من جهازك";
    }
  } else if (mode === "dataset") {
    if (selectedFpPath) {
      preview.src = "/" + selectedFpPath;
      preview.style.display = "block";
      placeholder.style.display = "none";
    }
  }
}

// ---------------------------------------------------------------------------
// File Upload Handlers
// ---------------------------------------------------------------------------

function handleFaceFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFaceFile = file;
  capturedFaceBlob = null; // Clear live capture if any

  const labelEl = document.getElementById("facePickedName");
  labelEl.innerText = `ملف محدد: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  labelEl.style.display = "inline-block";

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    const imgFace = document.getElementById("presetFaceImg");
    imgFace.src = e.target.result;
    imgFace.style.display = "block";
    document.getElementById("camPlaceholder").style.display = "none";
    document.getElementById("webcamVideo").style.display = "none";
  };
  reader.readAsDataURL(file);
}

function handleVoiceFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  uploadedVoiceFile = file;
  recordedAudioBlob = null; // Clear mic record

  const labelEl = document.getElementById("voicePickedName");
  labelEl.innerText = `ملف صوتي: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  labelEl.style.display = "inline-block";
  document.getElementById("voiceStatusText").innerText = `تم تحميل الملف: ${file.name}`;
}

function handleFpFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  uploadedFpFile = file;

  const labelEl = document.getElementById("fpPickedName");
  labelEl.innerText = `ملف بصمة: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  labelEl.style.display = "inline-block";

  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById("fpPreviewImg");
    preview.src = e.target.result;
    preview.style.display = "block";
    document.getElementById("fpPlaceholder").style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------------------
// Dropdown Selection Handlers
// ---------------------------------------------------------------------------

function handleFaceDropdownSelect(selectEl) {
  selectedFacePath = selectEl.value;
  if (selectedFacePath) {
    const imgFace = document.getElementById("presetFaceImg");
    imgFace.src = "/" + selectedFacePath;
    imgFace.style.display = "block";
    document.getElementById("camPlaceholder").style.display = "none";
    document.getElementById("webcamVideo").style.display = "none";
  }
}

function handleVoiceDropdownSelect(selectEl) {
  selectedVoicePath = selectEl.value;
  if (selectedVoicePath) {
    document.getElementById("voiceStatusText").innerText = `عينة داتا: ${selectedVoicePath.split("/").pop()}`;
  }
}

function handleFpDropdownSelect(selectEl) {
  selectedFpPath = selectEl.value;
  if (selectedFpPath) {
    const preview = document.getElementById("fpPreviewImg");
    preview.src = "/" + selectedFpPath;
    preview.style.display = "block";
    document.getElementById("fpPlaceholder").style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// Webcam & Face Live Stream
// ---------------------------------------------------------------------------

async function toggleCamera() {
  const video = document.getElementById("webcamVideo");
  const placeholder = document.getElementById("camPlaceholder");
  const btnStart = document.getElementById("btnStartCamera");
  const btnCap = document.getElementById("btnCaptureFace");
  const radar = document.getElementById("radarLine");

  if (webcamStream) {
    stopCamera();
  } else {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" }
      });
      video.srcObject = webcamStream;
      video.style.display = "block";
      placeholder.style.display = "none";
      radar.style.display = "block";
      btnStart.innerHTML = "<span>⏹️</span> إيقاف الكاميرا";
      btnStart.classList.replace("btn-primary", "btn-secondary");
      btnCap.style.display = "inline-flex";
    } catch (err) {
      alert("تعذر فتح الكاميرا: " + err.message + "\nتأكد من إعطاء المتصفح إذن الكاميرا (Camera Permission) في شريط العنوان.");
    }
  }
}

function stopCamera() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const video = document.getElementById("webcamVideo");
  const btnStart = document.getElementById("btnStartCamera");
  const btnCap = document.getElementById("btnCaptureFace");
  const radar = document.getElementById("radarLine");

  if (video) video.style.display = "none";
  if (radar) radar.style.display = "none";
  if (btnStart) {
    btnStart.innerHTML = "<span>📹</span> تشغيل الكاميرا";
    btnStart.classList.replace("btn-secondary", "btn-primary");
  }
  if (btnCap) btnCap.style.display = "none";
}

function captureFaceSnapshot() {
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("capturedFaceCanvas");
  if (!webcamStream) {
    alert("يرجى تشغيل الكاميرا أولاً!");
    return;
  }

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    capturedFaceBlob = blob;
    uploadedFaceFile = null;
    canvas.style.display = "block";
    video.style.display = "none";
    stopCamera();
    document.getElementById("camPlaceholder").style.display = "none";
    alert("تم التقاط لقطة الوجه بنجاح!");
  }, "image/jpeg", 0.95);
}

// ---------------------------------------------------------------------------
// Microphone & Audio Recording
// ---------------------------------------------------------------------------

function setupAudioVisualizer() {
  const canvas = document.getElementById("voiceWaveCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.parentElement.clientWidth || 300;
  canvas.height = 90;

  function drawIdle() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(0, 229, 255, 0.35)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }
  drawIdle();
}

async function toggleAudioRecording() {
  const btn = document.getElementById("btnRecordVoice");
  const statusTxt = document.getElementById("voiceStatusText");

  if (audioRecorder && audioRecorder.state === "recording") {
    audioRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 256;
    source.connect(analyserNode);

    audioChunks = [];
    audioRecorder = new MediaRecorder(stream);

    audioRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    audioRecorder.onstop = async () => {
      try {
        const rawBlob = new Blob(audioChunks, { type: audioRecorder.mimeType || "audio/webm" });
        const arrayBuffer = await rawBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        recordedAudioBlob = encodeWAV(audioBuffer, 16000);
        uploadedVoiceFile = null;
        statusTxt.innerText = "تم تسجيل صوتك وتحويله لصيغة 16kHz WAV بنجاح! جاهز للفحص.";
      } catch (e) {
        console.warn("WAV conversion fallback:", e);
        recordedAudioBlob = new Blob(audioChunks, { type: "audio/wav" });
        statusTxt.innerText = "تم تسجيل الصوت بنجاح! جاهز للفحص.";
      }

      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(waveAnimFrame);
      btn.innerHTML = "<span>🎤</span> تسجيل صوتي (2.5 ثانية)";
      btn.classList.replace("btn-danger", "btn-primary");
    };

    audioRecorder.start();
    btn.innerHTML = "<span>⏹️</span> جاري التسجيل... تكلم الآن";
    btn.classList.replace("btn-primary", "btn-danger");
    statusTxt.innerText = "تسجيل حي: انطق 'Open security gate now'...";

    renderWaveform();

    setTimeout(() => {
      if (audioRecorder && audioRecorder.state === "recording") {
        audioRecorder.stop();
      }
    }, 2500);
  } catch (err) {
    alert("تعذر الوصول إلى المايكروفون: " + err.message + "\nتأكد من إعطاء إذن المايك في المتصفح.");
  }
}

function renderWaveform() {
  const canvas = document.getElementById("voiceWaveCanvas");
  const ctx = canvas.getContext("2d");
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    waveAnimFrame = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "#05080e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00e5ff";
    ctx.beginPath();

    const sliceWidth = (canvas.width * 1.0) / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }
  draw();
}

/**
 * Encodes an AudioBuffer into pure 16-bit 16kHz Mono PCM WAV Blob.
 */
function encodeWAV(audioBuffer, targetSampleRate = 16000) {
  const channelData = audioBuffer.getChannelData(0);
  let samples;

  if (audioBuffer.sampleRate !== targetSampleRate) {
    const ratio = audioBuffer.sampleRate / targetSampleRate;
    const newLength = Math.round(channelData.length / ratio);
    samples = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      samples[i] = channelData[Math.min(Math.round(i * ratio), channelData.length - 1)];
    }
  } else {
    samples = channelData;
  }

  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono channel
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); // 16-bit
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

// ---------------------------------------------------------------------------
// Individual Modality Verifications (Mode A, Mode B, Fingerprint)
// ---------------------------------------------------------------------------

async function verifySingleModality(modality) {
  const formData = new FormData();
  formData.append("user_id", claimedUser);

  let endpoint = "";

  if (modality === "face") {
    endpoint = "/api/verify/face";
    if (faceInputMode === "live") {
      if (!capturedFaceBlob) {
        alert("يرجى تشغيل الكاميرا والتقاط لقطة أولاً (Capture Snapshot).");
        return;
      }
      formData.append("image", capturedFaceBlob, "face.jpg");
    } else if (faceInputMode === "upload") {
      if (!uploadedFaceFile) {
        alert("يرجى اختيار ملف صورة من جهازك أولاً.");
        return;
      }
      formData.append("image", uploadedFaceFile);
    } else {
      if (!selectedFacePath) {
        alert("يرجى اختيار عينة من قائمة الداتا.");
        return;
      }
      formData.append("preset_path", selectedFacePath);
    }
  } else if (modality === "voice") {
    endpoint = "/api/verify/voice";
    if (voiceInputMode === "live") {
      if (!recordedAudioBlob) {
        alert("يرجى الضغط على زر التسجيل ونطق العبارة أولاً.");
        return;
      }
      formData.append("audio", recordedAudioBlob, "voice.wav");
    } else if (voiceInputMode === "upload") {
      if (!uploadedVoiceFile) {
        alert("يرجى اختيار ملف صوتي من جهازك أولاً.");
        return;
      }
      formData.append("audio", uploadedVoiceFile);
    } else {
      if (!selectedVoicePath) {
        alert("يرجى اختيار عينة صوتية من قائمة الداتا.");
        return;
      }
      formData.append("preset_path", selectedVoicePath);
    }
  } else if (modality === "fingerprint") {
    endpoint = "/api/verify/fingerprint";
    if (fpInputMode === "upload") {
      if (!uploadedFpFile) {
        alert("يرجى اختيار ملف صورة بصمة من جهازك أولاً.");
        return;
      }
      formData.append("image", uploadedFpFile);
    } else {
      if (!selectedFpPath) {
        alert("يرجى اختيار عينة بصمة من قائمة الداتا.");
        return;
      }
      formData.append("preset_path", selectedFpPath);
    }
  }

  try {
    const res = await fetch(endpoint, { method: "POST", body: formData });
    const result = await res.json();

    if (!res.ok) {
      alert("خطأ أثناء الفحص: " + (result.detail || res.statusText));
      return;
    }

    displaySingleResult(modality, result);
  } catch (err) {
    alert("تعذر الاتصال بالسيرفر: " + err.message);
  }
}

function displaySingleResult(modality, res) {
  const isVerified = res.verified;
  const scorePct = res.score_percent;
  const timeMs = res.time_ms;

  if (modality === "face") {
    const card = document.getElementById("stageFaceCard");
    const badge = document.getElementById("badgeFace");
    card.className = "stage-card " + (isVerified ? "passed" : "failed");
    badge.className = "stage-badge " + (isVerified ? "passed" : "failed");
    badge.innerText = isVerified ? `مقبول (${scorePct}%)` : `مرفوض (${scorePct}%)`;

    document.getElementById("faceScoreVal").innerText = scorePct + "%";
    document.getElementById("faceBarFill").style.width = scorePct + "%";
    document.getElementById("faceBarFill").className = "metric-bar-fill " + (isVerified ? "passed" : "failed");
    document.getElementById("timeFace").innerText = timeMs + " ms";
  } else if (modality === "voice") {
    const card = document.getElementById("stageVoiceCard");
    const badge = document.getElementById("badgeVoice");
    card.className = "stage-card " + (isVerified ? "passed" : "failed");
    badge.className = "stage-badge " + (isVerified ? "passed" : "failed");
    badge.innerText = isVerified ? `مقبول (${scorePct}%)` : `مرفوض (${scorePct}%)`;

    document.getElementById("voiceScoreVal").innerText = scorePct + "%";
    document.getElementById("voiceBarFill").style.width = scorePct + "%";
    document.getElementById("voiceBarFill").className = "metric-bar-fill " + (isVerified ? "passed" : "failed");
    document.getElementById("timeVoice").innerText = timeMs + " ms";
  } else if (modality === "fingerprint") {
    const card = document.getElementById("stageFpCard");
    const badge = document.getElementById("badgeFp");
    card.className = "stage-card " + (isVerified ? "passed" : "failed");
    badge.className = "stage-badge " + (isVerified ? "passed" : "failed");
    badge.innerText = isVerified ? `مقبول (${scorePct}%)` : `مرفوض (${scorePct}%)`;

    document.getElementById("fpScoreVal").innerText = scorePct + "%";
    document.getElementById("fpBarFill").style.width = scorePct + "%";
    document.getElementById("fpBarFill").className = "metric-bar-fill " + (isVerified ? "passed" : "failed");
    document.getElementById("timeFp").innerText = timeMs + " ms";
  }

  const banner = document.getElementById("decisionBanner");
  banner.className = "decision-banner " + (isVerified ? "granted" : "denied");
  document.getElementById("decisionTitle").innerText = isVerified ? "MODALITY VERIFIED" : "MODALITY REJECTED";
  document.getElementById("decisionDesc").innerText = `فحص نمط ${modality.toUpperCase()} للمستخدم ${res.user_id}: نسبة التطابق ${scorePct}%`;
}

// ---------------------------------------------------------------------------
// 1-Click Presentation Presets
// ---------------------------------------------------------------------------

async function runPresetDemo(demoType) {
  resetVerificationState();

  if (demoType === "member1_genuine") {
    claimedUser = "member1";
    document.querySelectorAll(".user-pill").forEach((p, idx) => p.classList.toggle("active", idx === 0));
    selectedFacePath = presetsData.member1.face_sample;
    selectedVoicePath = presetsData.member1.voice_sample;
    selectedFpPath = presetsData.member1.fp_sample;
  } else if (demoType === "member2_genuine") {
    claimedUser = "member2";
    document.querySelectorAll(".user-pill").forEach((p, idx) => p.classList.toggle("active", idx === 1));
    selectedFacePath = presetsData.member2.face_sample;
    selectedVoicePath = presetsData.member2.voice_sample;
    selectedFpPath = presetsData.member2.fp_sample;
  } else if (demoType === "impostor_attack") {
    claimedUser = "member1"; // Attack member 1's identity
    document.querySelectorAll(".user-pill").forEach((p, idx) => p.classList.toggle("active", idx === 0));
    selectedFacePath = presetsData.impostor.face_sample;
    selectedVoicePath = presetsData.impostor.voice_sample;
    selectedFpPath = presetsData.impostor.fp_sample;
  } else if (demoType === "spoof_voice") {
    claimedUser = "member1";
    document.querySelectorAll(".user-pill").forEach((p, idx) => p.classList.toggle("active", idx === 0));
    selectedFacePath = presetsData.member1.face_sample; // right face
    selectedVoicePath = presetsData.impostor.voice_sample; // wrong voice!
    selectedFpPath = presetsData.member1.fp_sample;
  }

  // Switch tabs to dataset mode for clear visual feedback
  setFaceInputMode("dataset");
  setVoiceInputMode("dataset");
  setFpInputMode("dataset");

  document.getElementById("faceSelectDropdown").value = selectedFacePath;
  document.getElementById("voiceSelectDropdown").value = selectedVoicePath;
  document.getElementById("fpSelectDropdown").value = selectedFpPath;

  handleFaceDropdownSelect(document.getElementById("faceSelectDropdown"));
  handleVoiceDropdownSelect(document.getElementById("voiceSelectDropdown"));
  handleFpDropdownSelect(document.getElementById("fpSelectDropdown"));

  await executeFullCascade();
}

// ---------------------------------------------------------------------------
// Full Cascade Mode D Verification
// ---------------------------------------------------------------------------

async function executeFullCascade() {
  const btn = document.getElementById("btnRunCascade");
  btn.disabled = true;
  btn.innerText = "⏳ جاري تنفيذ الفحص الهرمي المتتابع (Cascade Mode D)...";

  const formData = new FormData();
  formData.append("user_id", claimedUser);

  // 1. Resolve Face
  if (faceInputMode === "live") {
    if (!capturedFaceBlob) {
      alert("تنبيه: الكاميرا الحية محددة ولكن لم يتم التقاط لقطة بعد! يرجى تشغيل الكاميرا والضغط على 'التقاط لقطة'، أو التبديل إلى 'اختيار من الداتا'.");
      btn.disabled = false;
      btn.innerText = "🚀 تنفيذ الفحص الهرمي المتتابع (Cascade Mode D)";
      return;
    }
    formData.append("face_image", capturedFaceBlob, "face.jpg");
  } else if (faceInputMode === "upload") {
    if (!uploadedFaceFile) {
      alert("تنبيه: يرجى رفع صورة الوجه من جهازك أولاً.");
      btn.disabled = false;
      btn.innerText = "🚀 تنفيذ الفحص الهرمي المتتابع (Cascade Mode D)";
      return;
    }
    formData.append("face_image", uploadedFaceFile);
  } else {
    if (!selectedFacePath) {
      selectedFacePath = presetsData[claimedUser]?.face_sample;
    }
    formData.append("face_preset", selectedFacePath);
  }

  // 2. Resolve Voice
  if (voiceInputMode === "live") {
    if (recordedAudioBlob) {
      formData.append("voice_audio", recordedAudioBlob, "voice.wav");
    } else {
      // Fallback to sample if not spoken
      formData.append("voice_preset", presetsData[claimedUser]?.voice_sample);
    }
  } else if (voiceInputMode === "upload") {
    if (uploadedVoiceFile) {
      formData.append("voice_audio", uploadedVoiceFile);
    } else {
      formData.append("voice_preset", presetsData[claimedUser]?.voice_sample);
    }
  } else {
    formData.append("voice_preset", selectedVoicePath || presetsData[claimedUser]?.voice_sample);
  }

  // 3. Resolve Fingerprint
  if (fpInputMode === "upload") {
    if (uploadedFpFile) {
      formData.append("fp_image", uploadedFpFile);
    } else {
      formData.append("fp_preset", presetsData[claimedUser]?.fp_sample);
    }
  } else {
    formData.append("fp_preset", selectedFpPath || presetsData[claimedUser]?.fp_sample);
  }

  try {
    const res = await fetch("/api/verify/cascade", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      alert("خطأ في التحقق: " + (data.detail || res.statusText));
      return;
    }

    displayCascadeResults(data);
  } catch (err) {
    alert("تعذر الاتصال بالسيرفر: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 تنفيذ الفحص الهرمي المتتابع (Cascade Mode D)";
  }
}

function displayCascadeResults(data) {
  const isGranted = data.decision;
  const lvl = data.level_reached;
  const scores = data.scores || {};
  const th = data.thresholds || {};
  const timing = data.time_per_stage || {};

  // Decision Banner
  const banner = document.getElementById("decisionBanner");
  const title = document.getElementById("decisionTitle");
  const desc = document.getElementById("decisionDesc");

  if (isGranted) {
    banner.className = "decision-banner granted";
    title.innerText = "ACCESS GRANTED [LEVEL " + lvl + "]";
    desc.innerText = `تم تأكيد الهوية بنجاح: ${data.user_id.toUpperCase()} (${data.role})`;
    document.getElementById("gateStatusText").innerText = "PERIMETER: UNLOCKED";
    document.getElementById("gateStatusBadge").style.borderColor = "var(--accent-green)";
    document.getElementById("gateStatusBadge").style.color = "var(--accent-green)";
    document.getElementById("dashboardUnlockBox").style.display = "block";

    currentSessionToken = data.session_token;
    currentRoleKey = data.role_key;
    document.getElementById("headerUserText").innerText = `${data.user_id} (${data.role})`;
  } else {
    banner.className = "decision-banner denied";
    title.innerText = "ACCESS DENIED";
    desc.innerText = `تم رفض الدخول وتوقف الفحص عند: ${data.stopped_at.toUpperCase()} (Level ${lvl})`;
    document.getElementById("gateStatusText").innerText = "PERIMETER: LOCKED";
    document.getElementById("gateStatusBadge").style.borderColor = "var(--accent-red)";
    document.getElementById("gateStatusBadge").style.color = "var(--accent-red)";
    document.getElementById("dashboardUnlockBox").style.display = "none";
  }

  // Stage Cards & Badges
  const cardFace = document.getElementById("stageFaceCard");
  const badgeFace = document.getElementById("badgeFace");
  const cardVoice = document.getElementById("stageVoiceCard");
  const badgeVoice = document.getElementById("badgeVoice");
  const cardFp = document.getElementById("stageFpCard");
  const badgeFp = document.getElementById("badgeFp");

  if (lvl >= 1 || (lvl === 0 && isGranted)) {
    cardFace.className = "stage-card passed";
    badgeFace.className = "stage-badge passed";
    badgeFace.innerText = "PASSED";
  } else {
    cardFace.className = "stage-card failed";
    badgeFace.className = "stage-badge failed";
    badgeFace.innerText = "FAILED / REJECTED";
  }

  if (lvl >= 2) {
    cardVoice.className = "stage-card passed";
    badgeVoice.className = "stage-badge passed";
    badgeVoice.innerText = "PASSED";
  } else if (lvl === 1) {
    cardVoice.className = "stage-card failed";
    badgeVoice.className = "stage-badge failed";
    badgeVoice.innerText = "FAILED / REJECTED";
  } else {
    cardVoice.className = "stage-card";
    badgeVoice.className = "stage-badge";
    badgeVoice.innerText = "BYPASSED (EARLY EXIT)";
  }

  if (lvl >= 3) {
    cardFp.className = "stage-card passed";
    badgeFp.className = "stage-badge passed";
    badgeFp.innerText = "PASSED (MAX LEVEL)";
  } else if (lvl === 2) {
    cardFp.className = "stage-card";
    badgeFp.className = "stage-badge";
    badgeFp.innerText = "OPTIONAL PASSED";
  } else {
    cardFp.className = "stage-card";
    badgeFp.className = "stage-badge";
    badgeFp.innerText = "SKIPPED";
  }

  // Modality Confidence Meters
  const faceScore = scores.face !== null ? scores.face : 0.0;
  const fusionScore = scores.fusion !== null ? scores.fusion : (scores.voice !== null ? scores.voice : 0.0);
  const fpScore = scores.fingerprint !== null ? scores.fingerprint : 0.0;

  document.getElementById("faceScoreVal").innerText = (faceScore * 100).toFixed(1) + "%";
  document.getElementById("faceBarFill").style.width = Math.min(100, faceScore * 100) + "%";
  document.getElementById("faceBarFill").className = "metric-bar-fill " + (faceScore >= (th.T_low || 0.5) ? "passed" : "failed");

  document.getElementById("voiceScoreVal").innerText = (fusionScore * 100).toFixed(1) + "%";
  document.getElementById("voiceBarFill").style.width = Math.min(100, fusionScore * 100) + "%";
  document.getElementById("voiceBarFill").className = "metric-bar-fill " + (fusionScore >= (th.T_fusion || 0.5) ? "passed" : "failed");

  document.getElementById("fpScoreVal").innerText = (fpScore * 100).toFixed(1) + "%";
  document.getElementById("fpBarFill").style.width = Math.min(100, fpScore * 100) + "%";
  document.getElementById("fpBarFill").className = "metric-bar-fill " + (fpScore >= (th.T_fp || 0.2) ? "passed" : "failed");

  // Threshold labels
  if (th.T_low) document.getElementById("tLowVal").innerText = (th.T_low * 100).toFixed(0) + "%";
  if (th.T_fusion) document.getElementById("tFusionVal").innerText = (th.T_fusion * 100).toFixed(0) + "%";
  if (th.T_fp) document.getElementById("tFpVal").innerText = (th.T_fp * 100).toFixed(0) + "%";

  // Latency Table
  document.getElementById("timeFace").innerText = timing.level1_face ? timing.level1_face.toFixed(1) + " ms" : "--";
  document.getElementById("exitFace").innerText = data.stopped_at === "level1" ? "YES (Stop)" : "Proceeded";

  document.getElementById("timeVoice").innerText = timing.level2_voice_fusion ? timing.level2_voice_fusion.toFixed(1) + " ms" : "--";
  document.getElementById("exitVoice").innerText = data.stopped_at === "level2" ? "YES (Stop)" : (timing.level2_voice_fusion ? "Proceeded" : "Skipped");

  document.getElementById("timeFp").innerText = timing.level3_fingerprint ? timing.level3_fingerprint.toFixed(1) + " ms" : "--";
  document.getElementById("exitFp").innerText = timing.level3_fingerprint ? "Completed" : "Skipped";

  document.getElementById("timeTotal").innerText = data.total_time_ms ? data.total_time_ms.toFixed(1) + " ms" : "--";

  // Refresh Dashboard Data in background
  fetchDashboardData();
}

// ---------------------------------------------------------------------------
// View Switcher (Gate vs Dashboard)
// ---------------------------------------------------------------------------

function switchView(viewName) {
  const gateView = document.getElementById("gateView");
  const dashView = document.getElementById("dashboardView");
  const tabGate = document.getElementById("tabGateBtn");
  const tabDash = document.getElementById("tabDashBtn");

  if (viewName === "dashboard") {
    gateView.style.display = "none";
    dashView.classList.add("active");
    tabGate.classList.remove("active");
    tabDash.classList.add("active");
    fetchDashboardData();
  } else {
    gateView.style.display = "grid";
    dashView.classList.remove("active");
    tabGate.classList.add("active");
    tabDash.classList.remove("active");
  }
}

// ---------------------------------------------------------------------------
// Operational Dashboard & Interactive Google-Sheets-Style Grid
// ---------------------------------------------------------------------------

async function fetchDashboardData() {
  try {
    const url = currentSessionToken ? `/api/dashboard/data?token=${currentSessionToken}` : `/api/dashboard/data`;
    const res = await fetch(url);
    const data = await res.json();

    allInventoryRows = data.inventory_sheet || [];
    renderDashboardUser(data.user);
    renderTelemetry(data.telemetry);
    renderSheetTable(allInventoryRows, data.user);
    renderAuditLogs(data.recent_audit_logs || []);
  } catch (err) {
    console.error("Error fetching dashboard data:", err);
  }
}

function renderDashboardUser(user) {
  const nameEl = document.getElementById("dashUserName");
  const roleEl = document.getElementById("dashRoleTitle");
  const pillEl = document.getElementById("sheetEditPermissionPill");
  const avatarEl = document.getElementById("dashAvatar");
  const vaultNotice = document.getElementById("vaultAccessNotice");
  const btnVault = document.getElementById("btnVaultUnlock");
  const btnEmerg = document.getElementById("btnEmergencyLock");

  if (user && user.authenticated) {
    nameEl.innerText = user.user_id === "member1" ? "Member 1 (Primary Admin)" : "Member 2 (Operator)";
    roleEl.innerText = `ROLE: ${user.role.toUpperCase()} (LEVEL ${user.level_reached})`;
    avatarEl.innerText = user.role_key === "super_admin" ? "👑" : "🛠️";

    if (user.can_edit_sheet) {
      pillEl.className = "system-status-pill";
      pillEl.style.borderColor = "var(--accent-green)";
      pillEl.style.color = "var(--accent-green)";
      pillEl.innerHTML = "✏️ Full Edit Mode Enabled (Super Admin)";
    } else {
      pillEl.className = "system-status-pill";
      pillEl.style.borderColor = "var(--accent-amber)";
      pillEl.style.color = "var(--accent-amber)";
      pillEl.innerHTML = "🔒 View-Only Mode (Operator)";
    }

    if (user.can_access_vault) {
      vaultNotice.innerHTML = "Access Authorized: <strong>Super Admin Key Validated</strong>.";
      btnVault.disabled = false;
      btnEmerg.disabled = false;
    } else {
      vaultNotice.innerHTML = "Restricted: <strong>Operator account lacks Core Vault clearance</strong>.";
      btnVault.disabled = true;
      btnEmerg.disabled = true;
    }
  } else {
    nameEl.innerText = "Guest / Unverified";
    roleEl.innerText = "ROLE: UNVERIFIED (READ ONLY)";
    pillEl.innerHTML = "🔒 View-Only Preview Mode";
    btnVault.disabled = true;
    btnEmerg.disabled = true;
  }
}

function renderTelemetry(telem) {
  if (!telem) return;
  document.getElementById("telemTemp").innerText = telem.temperature_c + " °C";
  document.getElementById("telemPower").innerText = telem.total_power_kw + " kW";
  document.getElementById("telemRacks").innerText = telem.active_nodes + " Racks";
  document.getElementById("telemUps").innerText = telem.ups_battery_pct + " %";
}

function renderSheetTable(rows, user) {
  const tbody = document.getElementById("inventoryTableBody");
  tbody.innerHTML = "";

  const canEdit = user && user.can_edit_sheet;

  rows.forEach(r => {
    const tr = document.createElement("tr");

    const statusBadgeClass = r.status === "OPTIMAL" ? "optimal" : "maintenance";
    const editBtnHtml = canEdit
      ? `<button class="btn-primary" style="padding: 4px 10px; font-size: 11px;" onclick="openEditModal('${r.id}')">✏️ Edit</button>`
      : `<span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">🔒 Locked</span>`;

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">${r.id}</td>
      <td style="font-family: var(--font-mono);">${r.rack}</td>
      <td><strong>${r.hostname}</strong></td>
      <td style="font-family: var(--font-mono);">${r.ip}</td>
      <td>${r.role}</td>
      <td style="font-family: var(--font-mono);">${r.power_kw} kW</td>
      <td style="font-family: var(--font-mono);">${r.temp_c} °C</td>
      <td><span class="badge-status ${statusBadgeClass}">${r.status}</span></td>
      <td style="color: var(--text-muted); font-size: 12px;">${r.notes || "--"}</td>
      <td>${editBtnHtml}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterSheetTable() {
  const query = document.getElementById("sheetSearchInput").value.toLowerCase().trim();
  const filtered = allInventoryRows.filter(r => {
    return (
      r.id.toLowerCase().includes(query) ||
      r.rack.toLowerCase().includes(query) ||
      r.hostname.toLowerCase().includes(query) ||
      r.ip.toLowerCase().includes(query) ||
      r.role.toLowerCase().includes(query) ||
      (r.notes && r.notes.toLowerCase().includes(query))
    );
  });

  const isSuperAdmin = currentRoleKey === "super_admin";
  renderSheetTable(filtered, { can_edit_sheet: isSuperAdmin });
}

function renderAuditLogs(logs) {
  const tbody = document.getElementById("auditTableBody");
  tbody.innerHTML = "";

  logs.forEach(row => {
    const tr = document.createElement("tr");
    const isGranted = row.decision === "ACCESS GRANTED";
    const statusCol = isGranted ? "color: var(--accent-green); font-weight: 700;" : "color: var(--accent-red); font-weight: 700;";

    tr.innerHTML = `
      <td style="font-family: var(--font-mono);">${row.timestamp}</td>
      <td><strong>${row.claimed_user.toUpperCase()}</strong></td>
      <td style="${statusCol}">${row.decision}</td>
      <td>Level ${row.level_reached}</td>
      <td style="font-family: var(--font-mono);">${(parseFloat(row.face_score) * 100 || 0).toFixed(1)}%</td>
      <td style="font-family: var(--font-mono);">${(parseFloat(row.voice_score) * 100 || 0).toFixed(1)}%</td>
      <td style="font-family: var(--font-mono);">${(parseFloat(row.fp_score) * 100 || 0).toFixed(1)}%</td>
      <td>${row.role}</td>
      <td style="font-family: var(--font-mono);">${row.time_ms} ms</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Sheet Edit Modal (Inline Editing for Super Admin)
// ---------------------------------------------------------------------------

function openEditModal(serverId) {
  const server = allInventoryRows.find(s => s.id === serverId);
  if (!server) return;

  document.getElementById("editServerId").value = server.id;
  document.getElementById("modalServerTitle").innerText = `Edit Server: ${server.id} (${server.hostname})`;
  document.getElementById("editServerRack").value = server.rack;
  document.getElementById("editServerHostname").value = server.hostname;
  document.getElementById("editServerIp").value = server.ip;
  document.getElementById("editServerRole").value = server.role;
  document.getElementById("editServerStatus").value = server.status;
  document.getElementById("editServerNotes").value = server.notes || "";

  document.getElementById("editModal").classList.add("active");
}

function closeEditModal() {
  document.getElementById("editModal").classList.remove("active");
}

async function saveServerEdit(event) {
  event.preventDefault();
  const id = document.getElementById("editServerId").value;
  const payload = {
    id: id,
    rack: document.getElementById("editServerRack").value,
    hostname: document.getElementById("editServerHostname").value,
    ip: document.getElementById("editServerIp").value,
    role: document.getElementById("editServerRole").value,
    status: document.getElementById("editServerStatus").value,
    notes: document.getElementById("editServerNotes").value,
  };

  try {
    const url = currentSessionToken ? `/api/dashboard/sheet/update?token=${currentSessionToken}` : `/api/dashboard/sheet/update`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert("Update blocked: " + (err.detail || res.statusText));
      return;
    }

    closeEditModal();
    await fetchDashboardData();
  } catch (err) {
    alert("Error updating sheet: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// Physical Gate & Vault Actions
// ---------------------------------------------------------------------------

async function executeGateAction(action) {
  if (!currentSessionToken) {
    alert("يرجى التحقق من هويتك عند البوابة أولاً!");
    switchView("gate");
    return;
  }

  const formData = new FormData();
  formData.append("action", action);

  try {
    const res = await fetch(`/api/dashboard/gate-action?token=${currentSessionToken}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      alert("تم رفض العملية: " + (data.detail || res.statusText));
      return;
    }

    alert(data.message);
    if (action === "perimeter_unlock") {
      document.getElementById("gateStatusText").innerText = "PERIMETER: UNLOCKED";
    } else if (action === "perimeter_lock") {
      document.getElementById("gateStatusText").innerText = "PERIMETER: LOCKED";
    }
  } catch (err) {
    alert("فشلت عملية البوابة: " + err.message);
  }
}
