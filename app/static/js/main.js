// SecureTranscribe Main JavaScript
class SecureTranscribe {
  constructor() {
    this.apiBase = "/api";
    this.currentTranscription = null;
    this.pollingInterval = null;
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.setupNavigation();
    this.startPolling();
    this.loadInitialData();
  }

  setupEventListeners() {
    // Form submission
    document.getElementById("uploadForm").addEventListener("submit", (e) => {
      e.preventDefault();
      this.handleFileUpload();
    });

    // File input change
    document.getElementById("audioFile").addEventListener("change", (e) => {
      this.validateFile(e.target.files[0]);
    });

    // Handle navigation clicks
    document.querySelectorAll('a[href^="/#"]').forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const targetId = link.getAttribute("href").substring(2); // Remove '/#'
        this.scrollToSection(targetId);
      });
    });

    // Drag and drop
    const uploadArea =
      document.querySelector(".upload-area") ||
      document.getElementById("audioFile").parentElement;
    if (uploadArea) {
      uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
      });

      uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
      });

      uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
          document.getElementById("audioFile").files = files;
          this.validateFile(files[0]);
        }
      });
    }
  }

  validateFile(file) {
    if (!file) return false;

    const validTypes = [
      "audio/mpeg",
      "audio/wav",
      "audio/mp4",
      "audio/flac",
      "audio/ogg",
    ];
    const maxSize = 500 * 1024 * 1024; // 500MB

    if (!validTypes.includes(file.type)) {
      this.showError(
        "Invalid file type. Please upload MP3, WAV, M4A, FLAC, or OGG files.",
      );
      return false;
    }

    if (file.size > maxSize) {
      this.showError("File too large. Maximum file size is 500MB.");
      return false;
    }

    return true;
  }

  async handleFileUpload() {
    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];

    if (!this.validateFile(file)) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", document.getElementById("language").value);
    formData.append("auto_start", document.getElementById("autoStart").checked);

    this.showUploadProgress(0, "Uploading file...");

    try {
      const response = await fetch(`${this.apiBase}/transcription/upload`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const result = await response.json();

      if (response.ok) {
        this.showSuccess(`File "${result.filename}" uploaded successfully!`);
        this.hideUploadProgress();
        fileInput.value = "";

        if (result.processing_started) {
          this.showInfo(
            "Transcription has been started and added to the queue.",
          );
        } else {
          this.showInfo("File uploaded. You can start transcription manually.");
        }

        this.refreshTranscriptions();
      } else {
        throw new Error(result.message || "Upload failed");
      }
    } catch (error) {
      this.showError(`Upload failed: ${error.message}`);
      this.hideUploadProgress();
    }
  }

  showUploadProgress(percentage, status) {
    const progressDiv = document.getElementById("uploadProgress");
    const progressBar = progressDiv.querySelector(".progress-bar");
    const statusText = document.getElementById("uploadStatus");

    progressDiv.style.display = "block";
    progressBar.style.width = `${percentage}%`;
    statusText.textContent = status;
  }

  hideUploadProgress() {
    document.getElementById("uploadProgress").style.display = "none";
  }

  async loadInitialData() {
    await this.refreshQueueStatus();
    await this.refreshTranscriptions();
  }

  async refreshQueueStatus() {
    try {
      const response = await fetch(`${this.apiBase}/queue/status`, {
        credentials: "include",
      });

      if (response.ok) {
        const status = await response.json();
        this.updateQueueDisplay(status);
      }
    } catch (error) {
      console.error("Failed to refresh queue status:", error);
    }
  }

  updateQueueDisplay(status) {
    document.getElementById("queuedJobs").textContent = status.queued_jobs || 0;
    document.getElementById("processingJobs").textContent =
      status.processing_jobs || 0;
    document.getElementById("completedJobs").textContent =
      status.completed_jobs || 0;
    document.getElementById("userPosition").textContent =
      status.user_queue_position ? `#${status.user_queue_position}` : "-";
  }

  async refreshTranscriptions() {
    try {
      const response = await fetch(`${this.apiBase}/transcription/list`, {
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        this.displayTranscriptions(data.transcriptions);
      }
    } catch (error) {
      console.error("Failed to refresh transcriptions:", error);
    }
  }

  displayTranscriptions(transcriptions) {
    const container = document.getElementById("transcriptionsList");

    if (transcriptions.length === 0) {
      container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-inbox fa-3x mb-3"></i>
                    <p>No transcriptions yet. Upload an audio file to get started.</p>
                </div>
            `;
      return;
    }

    const html = transcriptions
      .map((trans) => this.createTranscriptionCard(trans))
      .join("");
    container.innerHTML = html;
  }

  createTranscriptionCard(transcription) {
    const statusClass = this.getStatusClass(transcription.status);
    const statusText = this.getStatusText(transcription.status);

    return `
            <div class="transcription-item card mb-3" data-id="${transcription.id}">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <h6 class="card-title mb-1">
                                <i class="fas fa-file-audio me-2"></i>${transcription.original_filename}
                            </h6>
                            <div class="transcription-meta">
                                <span class="me-3">
                                    <i class="fas fa-clock me-1"></i>${transcription.formatted_duration}
                                </span>
                                <span class="me-3">
                                    <i class="fas fa-file me-1"></i>${transcription.file_format.toUpperCase()}
                                </span>
                                <span>
                                    <i class="fas fa-calendar me-1"></i>${new Date(transcription.created_at).toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                        <div class="col-md-3 text-center">
                            <span class="status-badge ${statusClass}">${statusText}</span>
                            ${
                              transcription.progress_percentage > 0
                                ? `
                                <div class="progress mt-2" style="height: 0.5rem;">
                                    <div class="progress-bar" style="width: ${transcription.progress_percentage}%"></div>
                                </div>
                                <small class="text-muted">${transcription.progress_percentage.toFixed(1)}%</small>
                            `
                                : ""
                            }
                        </div>
                        <div class="col-md-3 text-end">
                            ${this.createTranscriptionActions(transcription)}
                        </div>
                    </div>
                </div>
            </div>
        `;
  }

  createTranscriptionActions(transcription) {
    let actions = [];

    if (transcription.status === "pending") {
      actions.push(`
                <button class="btn btn-sm btn-primary" onclick="app.startTranscription(${transcription.id})">
                    <i class="fas fa-play me-1"></i>Start
                </button>
            `);
    }

    if (transcription.status === "completed") {
      actions.push(`
                <button class="btn btn-sm btn-success" onclick="app.viewTranscription(${transcription.id})">
                    <i class="fas fa-eye me-1"></i>View
                </button>
            `);
      actions.push(`
                <div class="btn-group me-2">
                    <button class="btn btn-sm btn-outline-primary dropdown-toggle" data-bs-toggle="dropdown">
                        <i class="fas fa-download me-1"></i>Export
                    </button>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item" href="#" onclick="app.exportTranscription(${transcription.id}, 'pdf')">PDF</a></li>
                        <li><a class="dropdown-item" href="#" onclick="app.exportTranscription(${transcription.id}, 'txt')">TXT</a></li>
                        <li><a class="dropdown-item" href="#" onclick="app.exportTranscription(${transcription.id}, 'csv')">CSV</a></li>
                        <li><a class="dropdown-item" href="#" onclick="app.exportTranscription(${transcription.id}, 'json')">JSON</a></li>
                    </ul>
                </div>
            `);
    }

    actions.push(`
            <button class="btn btn-sm btn-outline-danger" onclick="app.deleteTranscription(${transcription.id})">
                <i class="fas fa-trash me-1"></i>Delete
            </button>
        `);

    return actions.join(" ");
  }

  getStatusClass(status) {
    const statusClasses = {
      pending: "status-pending",
      processing: "status-processing",
      completed: "status-completed",
      failed: "status-failed",
    };
    return statusClasses[status] || "status-pending";
  }

  getStatusText(status) {
    const statusTexts = {
      pending: "Pending",
      processing: "Processing",
      completed: "Completed",
      failed: "Failed",
    };
    return statusTexts[status] || "Unknown";
  }

  async startTranscription(transcriptionId) {
    try {
      const response = await fetch(
        `${this.apiBase}/transcription/start/${transcriptionId}`,
        {
          method: "POST",
          credentials: "include",
        },
      );

      const result = await response.json();

      if (response.ok) {
        this.showSuccess("Transcription started successfully!");
        this.refreshTranscriptions();
      } else {
        throw new Error(result.message || "Failed to start transcription");
      }
    } catch (error) {
      this.showError(`Failed to start transcription: ${error.message}`);
    }
  }

  async viewTranscription(transcriptionId) {
    try {
      const response = await fetch(
        `${this.apiBase}/transcription/status/${transcriptionId}`,
        {
          credentials: "include",
        },
      );

      if (response.ok) {
        const transcription = await response.json();
        this.displayTranscriptionDetails(transcription);
      } else {
        throw new Error("Failed to load transcription details");
      }
    } catch (error) {
      this.showError(`Failed to load transcription: ${error.message}`);
    }
  }

  displayTranscriptionDetails(transcription) {
    const detailsDiv = document.getElementById("transcriptionDetails");
    const contentDiv = document.getElementById("transcriptionContent");

    let html = `
            <div class="row mb-3">
                <div class="col-md-6">
                    <h6>File Information</h6>
                    <p><strong>Name:</strong> ${transcription.file_info.filename}</p>
                    <p><strong>Duration:</strong> ${transcription.file_info.formatted_duration}</p>
                    <p><strong>Format:</strong> ${transcription.file_info.format.toUpperCase()}</p>
                    <p><strong>Language:</strong> ${transcription.language_detected || "Unknown"}</p>
                    <p><strong>Speakers:</strong> ${transcription.num_speakers}</p>
                    <p><strong>Confidence:</strong> ${(transcription.confidence_score * 100).toFixed(1)}%</p>
                </div>
                <div class="col-md-6">
                    <h6>Processing Information</h6>
                    <p><strong>Status:</strong> <span class="status-badge ${this.getStatusClass(transcription.status)}">${this.getStatusText(transcription.status)}</span></p>
                    <p><strong>Created:</strong> ${new Date(transcription.created_at).toLocaleString()}</p>
                    <p><strong>Completed:</strong> ${transcription.completed_at ? new Date(transcription.completed_at).toLocaleString() : "N/A"}</p>
                    <p><strong>Processing Time:</strong> ${transcription.processing_time ? transcription.processing_time.toFixed(1) + "s" : "N/A"}</p>
                </div>
            </div>
        `;

    if (transcription.segments && transcription.segments.length > 0) {
      html += '<h6>Transcript</h6><div class="transcript-content">';

      transcription.segments.forEach((segment) => {
        html += `
                    <div class="mb-3 p-3 border rounded">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <span class="speaker-label">${segment.speaker}</span>
                            <span class="timestamp">${this.formatTimestamp(segment.start_time)}</span>
                        </div>
                        <p class="mb-0">${segment.text}</p>
                    </div>
                `;
      });

      html += "</div>";
    } else if (transcription.full_transcript) {
      html += `
                <h6>Full Transcript</h6>
                <div class="transcript-content">
                    <div class="p-3 border rounded">
                        <p class="mb-0">${transcription.full_transcript}</p>
                    </div>
                </div>
            `;
    }

    contentDiv.innerHTML = html;
    detailsDiv.style.display = "block";
    detailsDiv.scrollIntoView({ behavior: "smooth" });
  }

  formatTimestamp(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }

  async exportTranscription(transcriptionId, format) {
    try {
      const response = await fetch(
        `${this.apiBase}/transcription/export/${transcriptionId}?export_format=${format}`,
        {
          credentials: "include",
        },
      );

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `transcription_${transcriptionId}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        this.showSuccess(`Exported as ${format.toUpperCase()} successfully!`);
      } else {
        throw new Error("Export failed");
      }
    } catch (error) {
      this.showError(`Export failed: ${error.message}`);
    }
  }

  async deleteTranscription(transcriptionId) {
    if (
      !confirm(
        "Are you sure you want to delete this transcription? This cannot be undone.",
      )
    ) {
      return;
    }

    try {
      const response = await fetch(
        `${this.apiBase}/transcription/${transcriptionId}`,
        {
          method: "DELETE",
          credentials: "include",
        },
      );

      const result = await response.json();

      if (response.ok) {
        this.showSuccess("Transcription deleted successfully!");
        this.refreshTranscriptions();

        // Hide details if viewing this transcription
        if (this.currentTranscription === transcriptionId) {
          document.getElementById("transcriptionDetails").style.display =
            "none";
          this.currentTranscription = null;
        }
      } else {
        throw new Error(result.message || "Delete failed");
      }
    } catch (error) {
      this.showError(`Failed to delete transcription: ${error.message}`);
    }
  }

  startPolling() {
    // Poll queue status every 5 seconds
    setInterval(() => {
      this.refreshQueueStatus();
    }, 5000);

    // Poll transcriptions every 10 seconds
    setInterval(() => {
      this.refreshTranscriptions();
    }, 10000);
  }

  // Toast notification methods
  showSuccess(message) {
    this.showToast("successToast", "successToastBody", message);
  }

  showError(message) {
    this.showToast("errorToast", "errorToastBody", message);
  }

  showInfo(message) {
    this.showToast("infoToast", "infoToastBody", message);
  }

  showToast(toastId, bodyId, message) {
    const toastElement = document.getElementById(toastId);
    const bodyElement = document.getElementById(bodyId);

    bodyElement.textContent = message;

    const toast = new bootstrap.Toast(toastElement);
    toast.show();
  }

  setupNavigation() {
    // Handle smooth scrolling to sections
    document.querySelectorAll('a[href^="/#"]').forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const targetId = link.getAttribute("href").substring(2); // Remove '/#'
        this.scrollToSection(targetId);
      });
    });

    // Handle initial hash in URL
    if (window.location.hash) {
      const targetId = window.location.hash.substring(1); // Remove '#'
      this.scrollToSection(targetId);
    }
  }

  scrollToSection(targetId) {
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      // Hide other sections temporarily to get accurate scroll
      const sections = [
        "upload",
        "queue",
        "transcriptions",
        "transcriptionDetails",
      ];
      sections.forEach((id) => {
        const elem = document.getElementById(id);
        if (elem && id !== targetId) {
          elem.style.display = "none";
        }
      });

      // Show target section
      targetElement.style.display = "block";
      targetElement.scrollIntoView({ behavior: "smooth", block: "start" });

      // Update active nav link
      document.querySelectorAll(".nav-link").forEach((link) => {
        link.classList.remove("active");
      });
      document
        .querySelector(`a[href="/#${targetId}"]`)
        ?.classList.add("active");
    } else {
      console.warn(`Section with ID "${targetId}" not found`);
      // If section not found, show default sections
      this.showDefaultSections();
    }
  }

  showDefaultSections() {
    // Show upload and queue by default
    const sections = ["upload", "queue", "transcriptions"];
    sections.forEach((id) => {
      const elem = document.getElementById(id);
      if (elem) {
        elem.style.display = "block";
      }
    });
    document.getElementById("transcriptionDetails").style.display = "none";
  }
}

// Global functions for inline event handlers
window.app = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.app = new SecureTranscribe();
});

// Utility functions
function showSessionInfo() {
  fetch("/api/sessions/current", { credentials: "include" })
    .then((response) => response.json())
    .then((session) => {
      // Handle undefined or NaN values gracefully
      const sessionId = session.session_id || "Unknown";
      const createdAt = session.created_at
        ? new Date(session.created_at).toLocaleString()
        : "Unknown";
      const filesProcessed = session.total_files_processed || 0;
      const avgConfidence = session.average_confidence
        ? (session.average_confidence * 100).toFixed(1)
        : "0.0";

      alert(
        `Session Information:\n\nSession ID: ${sessionId}\nCreated: ${createdAt}\nFiles Processed: ${filesProcessed}\nAverage Confidence: ${avgConfidence}%`,
      );
    })
    .catch((error) => {
      console.error("Failed to get session info:", error);
      alert("Failed to get session information");
    });
}

function refreshTranscriptions() {
  if (window.app) {
    window.app.refreshTranscriptions();
  }
}
