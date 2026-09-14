const API_BASE = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropZone = $("dropZone");
const browseBtn = $("browseBtn");
const selectedFile = $("selectedFile");
const fileName = $("fileName");
const fileSize = $("fileSize");
const removeFile = $("removeFile");

const urlInput = $("urlInput");
const pasteBtn = $("pasteBtn");
const analyzeBtn = $("analyzeBtn");
const languageSelect = $("languageSelect");

const filePanel = $("filePanel");
const urlPanel = $("urlPanel");
const sourceTabs = document.querySelectorAll(".source-tab");

const progressCard = $("progressCard");
const progressBar = $("progressBar");
const progressTitle = $("progressTitle");
const progressPercent = $("progressPercent");
const errorBox = $("errorBox");

const results = $("results");
const uploadWorkspace = $("uploadWorkspace");

const resultTitle = $("resultTitle");
const sessionId = $("sessionId");
const summaryContent = $("summaryContent");
const actionsContent = $("actionsContent");
const decisionsContent = $("decisionsContent");
const questionsContent = $("questionsContent");
const transcriptContent = $("transcriptContent");

const chatForm = $("chatForm");
const questionInput = $("questionInput");
const chatMessages = $("chatMessages");
const askBtn = $("askBtn");

let selectedMode = "file";
let selectedFileObject = null;
let currentAnalysisId = null;

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 2800);
}

function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 2 : 0)} ${units[index]}`;
}

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderText(text) {
  if (!text || !text.trim()) {
    return '<span class="muted">No information found.</span>';
  }

  return escapeHTML(text);
}

function setProgress(step) {
  const widths = [12, 35, 65, 88, 100];
  progressBar.style.width = `${widths[step] || 12}%`;

  const labels = [
    "Preparing media...",
    "Transcribing audio...",
    "Running AI analysis...",
    "Building knowledge base...",
    "Analysis complete!"
  ];

  progressTitle.textContent = labels[step] || labels[0];
  progressPercent.textContent = step === 4 ? "Done" : `${widths[step] || 12}%`;

  document.querySelectorAll(".step").forEach((el, index) => {
    el.classList.toggle("active", index <= step - 1);
  });
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error();

    $("apiText").textContent = "API Online";
    $("apiPill").querySelector(".status-dot").classList.add("online");
    $("sideStatus").classList.add("online");
  } catch {
    $("apiText").textContent = "API Offline";
    $("apiPill").querySelector(".status-dot").classList.add("offline");
    $("sideStatus").classList.add("offline");
  }
}

function selectFile(file) {
  if (!file) return;

  selectedFileObject = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);

  selectedFile.classList.remove("hidden");
  dropZone.classList.add("hidden");

  clearError();
}

browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  selectFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  selectFile(file);
});

removeFile.addEventListener("click", () => {
  selectedFileObject = null;
  fileInput.value = "";
  selectedFile.classList.add("hidden");
  dropZone.classList.remove("hidden");
});

sourceTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    sourceTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");

    selectedMode = tab.dataset.mode;

    filePanel.classList.toggle("hidden", selectedMode !== "file");
    urlPanel.classList.toggle("hidden", selectedMode !== "url");

    clearError();
  });
});

pasteBtn.addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    urlInput.value = text;
    showToast("URL pasted");
  } catch {
    showToast("Clipboard access was not available");
  }
});

function resetProgress() {
  progressCard.classList.add("hidden");
  progressBar.style.width = "8%";
}

function showProgress() {
  progressCard.classList.remove("hidden");
  setProgress(0);
}

async function analyze() {
  clearError();

  if (selectedMode === "file" && !selectedFileObject) {
    setError("Please choose a video or audio file first.");
    return;
  }

  if (selectedMode === "url" && !urlInput.value.trim()) {
    setError("Please paste a media URL first.");
    return;
  }

  analyzeBtn.disabled = true;
  showProgress();
  uploadWorkspace.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const formData = new FormData();
    formData.append("language", languageSelect.value);

    let endpoint;

    if (selectedMode === "file") {
      endpoint = "/api/analyze";
      formData.append("file", selectedFileObject);
    } else {
      endpoint = "/api/analyze-url";
      formData.append("url", urlInput.value.trim());
    }

    setProgress(1);

    const responsePromise = fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: formData
    });

    setProgress(2);

    const response = await responsePromise;

    setProgress(3);

    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    if (!response.ok || !payload.success) {
      throw new Error(payload.detail || "Analysis failed.");
    }

    setProgress(4);

    renderResults(payload.data);

    setTimeout(() => {
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 250);

    showToast("Video analysis completed");
  } catch (error) {
    console.error(error);
    resetProgress();
    setError(error.message || "Something went wrong.");
  } finally {
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyze);

function renderResults(data) {
  currentAnalysisId = data.analysis_id;

  resultTitle.textContent = data.title || "Untitled Video";
  sessionId.textContent = `Session: ${data.analysis_id || "N/A"}`;

  summaryContent.innerHTML = renderText(data.summary);
  actionsContent.innerHTML = renderText(data.action_items);
  decisionsContent.innerHTML = renderText(data.key_decisions);
  questionsContent.innerHTML = renderText(data.open_questions);

  transcriptContent.textContent = data.transcript || "";

  chatMessages.innerHTML = `
    <div class="chat-empty">
      <div class="chat-orb">✦</div>
      <p>Ask a question and I’ll search the transcript for the answer.</p>
    </div>
  `;

  results.classList.remove("hidden");
}

function addMessage(type, text) {
  const empty = chatMessages.querySelector(".chat-empty");
  if (empty) empty.remove();

  const row = document.createElement("div");
  row.className = `chat-message ${type}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.appendChild(bubble);
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return row;
}

async function askQuestion(question) {
  if (!currentAnalysisId) {
    showToast("Analyze a video first");
    return;
  }

  question = question.trim();
  if (!question) return;

  addMessage("user", question);

  questionInput.value = "";
  askBtn.disabled = true;

  const loading = addMessage("assistant", "Thinking...");

  try {
    const formData = new FormData();
    formData.append("analysis_id", currentAnalysisId);
    formData.append("question", question);

    const response = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      body: formData
    });

    const payload = await response.json();

    if (!response.ok || !payload.success) {
      throw new Error(payload.detail || "Could not answer the question.");
    }

    loading.querySelector(".bubble").textContent =
      payload.answer || "No answer returned.";
  } catch (error) {
    loading.querySelector(".bubble").textContent =
      `Error: ${error.message}`;
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(questionInput.value);
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => {
    askQuestion(button.textContent);
  });
});

$("copyTranscript").addEventListener("click", async () => {
  const text = transcriptContent.textContent;

  try {
    await navigator.clipboard.writeText(text);
    $("copyTranscript").textContent = "Copied!";
    setTimeout(() => $("copyTranscript").textContent = "Copy", 1500);
  } catch {
    showToast("Could not copy transcript");
  }
});

function resetWorkspace() {
  currentAnalysisId = null;
  selectedFileObject = null;
  fileInput.value = "";
  urlInput.value = "";

  selectedFile.classList.add("hidden");
  dropZone.classList.remove("hidden");
  results.classList.add("hidden");

  clearError();
  resetProgress();

  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("newBtn").addEventListener("click", resetWorkspace);
$("newAnalysisBtn").addEventListener("click", resetWorkspace);

$("healthBtn").addEventListener("click", async () => {
  await checkHealth();
  showToast($("apiText").textContent);
});

checkHealth();
