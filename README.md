# STT AI

**STT AI** is a powerful, offline-first Speech-to-Text application built for privacy and performance. It leverages state-of-the-art models like **SenseVoiceSmall**, **Paraformer-zh**, and **Fun-ASR-Nano** to provide accurate real-time transcription and file processing.

## Features

-   **Multi-Model Support**: Switch seamlessly between **SenseVoiceSmall**, **Paraformer-zh**, and **Fun-ASR-Nano** depending on your accuracy vs. speed needs.
-   **Offline Privacy**: All processing happens locally on your machine—no data leaves your device.
-   **Real-time Streaming**: Instant transcription as you speak with the streaming-capable Paraformer model.
-   **File Transcription**: Drag and drop audio/video files for fast batch processing.
-   **Inverse Text Normalization (ITN)**: Automatically formats numbers, dates, and punctuation (configurable).
-   **Model Management**: Download and manage models directly from the UI.
-   **Modern UI**: Built with React, Tailwind CSS, and Shadcn UI, supporting dark mode and a responsive design.

## Tech Stack

-   **Frontend**: [Tauri 2.0](https://tauri.app/), [React](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Tailwind CSS](https://tailwindcss.com/)
-   **Backend**: [Python](https://www.python.org/), [FastAPI](https://fastapi.tiangolo.com/), [FunASR](https://github.com/alibaba-damo-academy/FunASR), [PyTorch](https://pytorch.org/)

## Prerequisites

Before running the application, ensure you have the following installed:

1.  **Node.js** (v18 or higher recommended)
2.  **Rust & Cargo** (Required for building the Tauri frontend)
3.  **Python 3.12**
4.  **FFmpeg** (Recommended for broad audio format support)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AirSodaz/stt-app.git
cd stt-app
```

### 2. Backend Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Frontend Setup

Install Node.js dependencies:

```bash
cd ui
npm install
cd ..
```

## Usage

### Unified Development Start (Recommended)

To start both the backend server and the frontend client simultaneously with live reloading:

```bash
# From the root directory
python start_dev.py
```

### Manual Start

If you prefer to run components separately:

**Backend:**
```bash
# Terminal 1
python server.py
```

**Frontend:**
```bash
# Terminal 2
cd ui
npm run tauri dev
```

## Build for Production

To create a standalone executable:

```bash
cd ui
npm run tauri build
```

The build artifacts will be located in `ui/src-tauri/target/release/`.

## Troubleshooting

-   **Model Loading**: The first run might take a moment as models are initialized. Check the console if models fail to download.
-   **Port Conflicts**: The backend runs on port `8000`. Ensure this port is free.

## License

[MIT](LICENSE)
