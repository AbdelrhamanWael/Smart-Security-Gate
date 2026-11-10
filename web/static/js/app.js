/**
 * app.js - Biometric Security Gate & Google-Sheets-style RBAC Dashboard
 * Supports:
 *   - Webcam stream capture & Microphone recording.
 *   - Fast 1-click preset demos (Member 1, Member 2, Impostor Attacks).
 *   - Cascaded pipeline verification (Face L1 -> Voice L2+Fusion -> Fingerprint L3).
 *   - Role-Based Access Control (Super Admin vs. Operator).
 *   - Interactive Server Inventory Data Sheet with search, filter, and inline edit.
 */

// State
let claimedUser = "member1";
let currentSessionToken = null;
let currentRoleKey = "guest";
let presetsData = null;

let webcamStream = null;
let capturedFaceBlob = null;
let recordedAudioBlob = null;
let sampleFacePath = null;
let sampleVoicePath = null;
let sampleFpPath = null;

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
  initDefaultPresets();
  setupAudioVisualizer();
  await fetchDashboardData();
});

function startClock() {
  const clockEl = document.getElementById("clockDisplay");
  setInterval(() => {
    const d = new Date();
    clockEl.innerText = d.toTimeString().split(" ")[0];
  }, 1000);
}

// ---------------------------------------------------------------------------
// Presets & User Selection
// ---------------------------------------------------------------------------

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    presetsData = await res.json();
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function initDefaultPresets() {
  if (!presetsData) return;
  const user = presetsData[claimedUser] || presetsData["member1"];
  sampleFacePath = user.face_sample;
  sampleVoicePath = user.voice_sample;
  sampleFpPath = user.fp_sample;
  updateStageDisplays();
}

function selectClaimedUser(userId, btnEl) {
  claimedUser = userId;
  document.querySelectorAll(".user-pill").forEach(p => p.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");

  initDefaultPresets();
  resetVerificationState();
}

function updateStageDisplays() {
  // Update Face Preview image
  const imgFace = document.getElementById("presetFaceImg");
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("capturedFaceCanvas");

  if (!capturedFaceBlob && !webcamStream) {
    imgFace.src = "/" + sampleFacePath;
    imgFace.style.display = "block";
    video.style.display = "none";
    canvas.style.display = "none";
  }

  // Update Fingerprint Preview image
  const imgFp = document.getElementById("fpPreviewImg");
  if (sampleFpPath) {
    imgFp.src = "/" + sampleFpPath;
  }
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
// 1-Click Presentation Presets
// ---------------------------------------------------------------------------

async function runPresetDemo(demoType) {
  resetVerificationState();

  if (demoType === "member1_genuine") {
    claimedUser = "member1";
    document.querySelectorAll(".user-pill").forEach((p, idx) => {
      p.classList.toggle("active", idx === 0);
    });
    sampleFacePath = presetsData.member1.face_sample;
    sampleVoicePath = presetsData.member1.voice_sample;
    sampleFpPath = presetsData.member1.fp_sample;
  } else if (demoType === "member2_genuine") {
    claimedUser = "member2";
    document.querySelectorAll(".user-pill").forEach((p, idx) => {
      p.classList.toggle("active", idx === 1);
    });
    sampleFacePath = presetsData.member2.face_sample;
    sampleVoicePath = presetsData.member2.voice_sample;
    sampleFpPath = presetsData.member2.fp_sample;
  } else if (demoType === "impostor_attack") {
    claimedUser = "member1"; // Attack member 1's identity!
    document.querySelectorAll(".user-pill").forEach((p, idx) => {
      p.classList.toggle("active", idx === 0);
    });
    sampleFacePath = presetsData.impostor.face_sample;
    sampleVoicePath = presetsData.impostor.voice_sample;
    sampleFpPath = presetsData.impostor.fp_sample;
  } else if (demoType === "spoof_voice") {
    claimedUser = "member1";
    document.querySelectorAll(".user-pill").forEach((p, idx) => {
      p.classList.toggle("active", idx === 0);
    });
    sampleFacePath = presetsData.member1.face_sample; // genuine face
    sampleVoicePath = presetsData.impostor.voice_sample; // wrong voice!
    sampleFpPath = presetsData.member1.fp_sample;
  }

  // Clear any live blobs so presets are used
  capturedFaceBlob = null;
  recordedAudioBlob = null;
  updateStageDisplays();

  // Execute full cascade
  await executeFullCascade();
}

// ---------------------------------------------------------------------------
// Webcam & Face Capture
// ---------------------------------------------------------------------------

async function toggleCamera() {
  const video = document.getElementById("webcamVideo");
  const imgFace = document.getElementById("presetFaceImg");
  const canvas = document.getElementById("capturedFaceCanvas");
  const btn = document.getElementById("btnStartCamera");

  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    video.style.display = "none";
    imgFace.style.display = "block";
    btn.innerHTML = "<span>📹</span> Start Webcam";
  } else {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = webcamStream;
      video.style.display = "block";
      imgFace.style.display = "none";
      canvas.style.display = "none";
      btn.innerHTML = "<span>⏹️</span> Stop Camera";
    } catch (err) {
      alert("Could not access webcam: " + err.message);
    }
  }
}

function captureFaceSnapshot() {
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("capturedFaceCanvas");
  if (!webcamStream) {
    alert("Please start the webcam first!");
    return;
  }

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    capturedFaceBlob = blob;
    canvas.style.display = "block";
    video.style.display = "none";
    if (webcamStream) {
      webcamStream.getTracks().forEach(t => t.stop());
      webcamStream = null;
      document.getElementById("btnStartCamera").innerHTML = "<span>📹</span> Start Webcam";
    }
  }, "image/jpeg", 0.95);
}

function loadSampleFace() {
  capturedFaceBlob = null;
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    document.getElementById("btnStartCamera").innerHTML = "<span>📹</span> Start Webcam";
  }
  const user = presetsData[claimedUser] || presetsData["member1"];
  sampleFacePath = user.face_sample;
  updateStageDisplays();
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

    audioRecorder.onstop = () => {
      recordedAudioBlob = new Blob(audioChunks, { type: "audio/wav" });
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(waveAnimFrame);
      btn.innerHTML = "<span>🎤</span> Record Passphrase (2.5s)";
      statusTxt.innerText = "Voice captured! Ready for verification.";
    };

    audioRecorder.start();
    btn.innerHTML = "<span>⏹️</span> Recording... Speak Now";
    statusTxt.innerText = "RECORDING: 'Open security gate now'...";

    renderWaveform();

    // Auto-stop after 2.5 seconds
    setTimeout(() => {
      if (audioRecorder && audioRecorder.state === "recording") {
        audioRecorder.stop();
      }
    }, 2500);
  } catch (err) {
    alert("Could not access microphone: " + err.message);
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

function loadSampleVoice() {
  recordedAudioBlob = null;
  const user = presetsData[claimedUser] || presetsData["member1"];
  sampleVoicePath = user.voice_sample;
  document.getElementById("voiceStatusText").innerText = "Sample voice loaded: " + sampleVoicePath.split("/").pop();
}

function loadSampleFingerprint() {
  const user = presetsData[claimedUser] || presetsData["member1"];
  sampleFpPath = user.fp_sample;
  document.getElementById("fpPreviewImg").src = "/" + sampleFpPath;
}

// ---------------------------------------------------------------------------
// Full Cascade Mode D Verification
// ---------------------------------------------------------------------------

async function executeFullCascade() {
  const btn = document.getElementById("btnRunCascade");
  btn.disabled = true;
  btn.innerText = "⏳ Executing Cascade Pipeline...";

  const formData = new FormData();
  formData.append("user_id", claimedUser);

  // Face
  if (capturedFaceBlob) {
    formData.append("face_image", capturedFaceBlob, "capture.jpg");
  } else {
    formData.append("face_preset", sampleFacePath);
  }

  // Voice
  if (recordedAudioBlob) {
    formData.append("voice_audio", recordedAudioBlob, "voice.wav");
  } else {
    formData.append("voice_preset", sampleVoicePath);
  }

  // Fingerprint
  formData.append("fp_preset", sampleFpPath);

  try {
    const res = await fetch("/api/verify/cascade", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    displayCascadeResults(data);
  } catch (err) {
    alert("Verification request failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 Execute Hierarchical Verification (Cascade Mode D)";
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
    desc.innerText = `Identity Confirmed: ${data.user_id.toUpperCase()} (${data.role})`;
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
    desc.innerText = `Verification stopped at: ${data.stopped_at.toUpperCase()} (Level ${lvl})`;
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
    alert("Please verify your identity at the Biometric Gate first!");
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
      alert("Action Denied: " + (data.detail || res.statusText));
      return;
    }

    alert(data.message);
    if (action === "perimeter_unlock") {
      document.getElementById("gateStatusText").innerText = "PERIMETER: UNLOCKED";
    } else if (action === "perimeter_lock") {
      document.getElementById("gateStatusText").innerText = "PERIMETER: LOCKED";
    }
  } catch (err) {
    alert("Gate action failed: " + err.message);
  }
}

