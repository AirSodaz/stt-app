import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { AudioInput } from './components/AudioInput';
import { Bot, Copy, Check, Settings2, Settings } from 'lucide-react';
import { cn } from './lib/utils';
import { SettingsPage } from './components/SettingsPage';
import { useStreamingASR } from './hooks/useStreamingASR';
import { logger } from './lib/logger';

function App() {
    const [transcription, setTranscription] = useState("");
    const [isProcessing, setIsProcessing] = useState(false);
    const [copied, setCopied] = useState(false);
    const [language, setLanguage] = useState("auto");
    const [useItn, setUseItn] = useState(false);
    // Model is null until we load preference or user selects one
    const [model, setModel] = useState<string | null>(null);
    const [availableModels, setAvailableModels] = useState<{ name: string, downloaded: boolean }[]>([]);
    const [downloadedModels, setDownloadedModels] = useState<string[]>([]);
    const [isDownloading, setIsDownloading] = useState<string | null>(null);
    const [downloadProgress, setDownloadProgress] = useState<number>(0);

    const [downloadMessage, setDownloadMessage] = useState<string>("");
    const [showSettings, setShowSettings] = useState(false);

    // Server connection state
    const [isLoadingModels, setIsLoadingModels] = useState(true);
    const [serverStatus, setServerStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');

    // Model loading state: 'idle' = no model loaded, 'loading' = loading in progress, 'ready' = model ready
    const [modelLoadingStatus, setModelLoadingStatus] = useState<'idle' | 'loading' | 'ready'>('idle');

    // Streaming state
    const [streamingText, setStreamingText] = useState("");

    // Processing time tracking (in seconds)
    const [processingTime, setProcessingTime] = useState<number>(0);

    // Callback for streaming partial results
    const handlePartialResult = useCallback((text: string, isFinal: boolean) => {
        if (text) {
            // Server sends accumulated text, so just replace
            setTranscription(text);
        }
        if (isFinal) {
            setStreamingText("");
            setIsProcessing(false);
        }
    }, []);

    // Initialize streaming hook
    const { isStreaming, startStreaming, stopStreaming } = useStreamingASR(handlePartialResult);


    const fetchModels = useCallback(async (retryCount = 0) => {
        try {
            if (retryCount === 0) {
                setIsLoadingModels(true);
                setServerStatus('connecting');
            }

            const res = await axios.get("http://127.0.0.1:8000/models", { timeout: 2000 });

            if (res.data && Array.isArray(res.data)) {
                setAvailableModels(res.data);
                // Extract downloaded models
                const downloaded = res.data.filter((m: any) => m.downloaded).map((m: any) => m.name);
                setDownloadedModels(downloaded);

                setServerStatus('connected');
                setIsLoadingModels(false);
                return true;
            }
        } catch (err) {
            logger.warn(`Failed to fetch models (attempt ${retryCount + 1}):`, err);

            // Retry on network errors or timeouts (likely server starting up)
            // Max retries: 30 attempts * 1s = 30s wait time
            if (retryCount < 30) {
                setTimeout(() => fetchModels(retryCount + 1), 1000);
            } else {
                setServerStatus('error');
                setIsLoadingModels(false);
                setAvailableModels([]);
                setDownloadedModels([]);
            }
            return false;
        }
    }, []);

    const loadModelPreference = useCallback(async () => {
        try {
            const res = await axios.get("http://127.0.0.1:8000/preference/model");
            if (res.data?.model) {
                setModel(res.data.model);
                // Check if this model is already loaded
                const statusRes = await axios.get("http://127.0.0.1:8000/models/status");
                if (statusRes.data?.loaded_model === res.data.model) {
                    setModelLoadingStatus('ready');
                } else if (statusRes.data?.is_loading && statusRes.data?.loading_model === res.data.model) {
                    setModelLoadingStatus('loading');
                    // Start polling for completion
                    pollModelStatus(res.data.model);
                } else {
                    setModelLoadingStatus('idle');
                }
            }
        } catch (err) {
            logger.error("Failed to load model preference:", err);
        }
    }, []);

    const saveModelPreference = useCallback(async (modelName: string) => {
        try {
            const formData = new FormData();
            formData.append("model_name", modelName);
            const res = await axios.post("http://127.0.0.1:8000/preference/model", formData);
            return res.data;
        } catch (err) {
            logger.error("Failed to save model preference:", err);
            return null;
        }
    }, []);

    // Poll model status until loaded
    const pollModelStatus = useCallback(async (targetModel: string) => {
        const maxAttempts = 120; // 2 minutes max
        let attempts = 0;

        const poll = async () => {
            try {
                const res = await axios.get("http://127.0.0.1:8000/models/status");
                const { loaded_model, is_loading, error } = res.data;

                if (error) {
                    logger.error("Model loading error:", error);
                    setModelLoadingStatus('idle');
                    return;
                }

                if (loaded_model === targetModel) {
                    setModelLoadingStatus('ready');
                    logger.debug(`Model ${targetModel} is now ready`);
                    return;
                }

                if (is_loading && attempts < maxAttempts) {
                    attempts++;
                    setTimeout(poll, 1000);
                } else if (!is_loading && loaded_model !== targetModel) {
                    // Loading finished but not for our model - might have been cancelled
                    setModelLoadingStatus('idle');
                }
            } catch (err) {
                logger.error("Failed to poll model status:", err);
                setModelLoadingStatus('idle');
            }
        };

        poll();
    }, []);

    useEffect(() => {
        const init = async () => {
            // Start fetching models (with retry)
            // Once connected, load request preference
            const success = await fetchModels();
            if (success) {
                await loadModelPreference();
            } else {
                // If fetchModels returns true immediately (which it won't due to async/recursive nature),
                // we might miss this. 
                // However, fetchModels is recursive via setTimeout, so we can't await the whole chain easily here.
                // Better approach: Just kick off fetchModels, and let loadModelPreference run independently 
                // or check serverStatus. But loadModelPreference also needs server.

                // Let's just try loading preference anyway, it might fail if server down, which is fine.
                // Or better, only load preference if we suspect server is up. 
                // But simplified: Just run init in parallel.
                loadModelPreference();
            }
        };
        init();
    }, [fetchModels, loadModelPreference]);

    useEffect(() => {
        // Polling to keep model status fresh (e.g. if external download)
        // Only poll if we are connected
        const interval = setInterval(() => {
            if (serverStatus === 'connected' && !isDownloading) {
                // silent update
                axios.get("http://127.0.0.1:8000/models").then(res => {
                    if (res.data && Array.isArray(res.data)) {
                        setAvailableModels(res.data);
                        const downloaded = res.data.filter((m: any) => m.downloaded).map((m: any) => m.name);
                        setDownloadedModels(downloaded);
                    }
                }).catch(() => {
                    // If polling fails, maybe server died? 
                    // setServerStatus('error'); 
                });
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [serverStatus, isDownloading]);

    const handleModelChange = async (newModel: string) => {
        setModel(newModel);
        setModelLoadingStatus('loading');

        const result = await saveModelPreference(newModel);

        if (result?.status === 'ready') {
            // Model was already loaded
            setModelLoadingStatus('ready');
        } else if (result?.status === 'loading') {
            // Model is loading in background, poll for completion
            pollModelStatus(newModel);
        } else {
            // Error or unexpected
            setModelLoadingStatus('idle');
        }
    };

    const handleDownloadModel = async (modelName: string) => {
        setIsDownloading(modelName);
        setDownloadProgress(0);
        setDownloadMessage("Starting download...");

        try {
            // Use SSE for progress tracking
            const eventSource = new EventSource(`http://127.0.0.1:8000/download_model/${modelName}`);

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setDownloadProgress(data.progress || 0);
                    setDownloadMessage(data.message || "");

                    if (data.status === "complete") {
                        eventSource.close();
                        fetchModels();
                        setModel(modelName);
                        saveModelPreference(modelName);
                        setIsDownloading(null);
                        setDownloadProgress(0);
                        setDownloadMessage("");
                    } else if (data.status === "error") {
                        eventSource.close();
                        alert(`Download failed: ${data.message}`);
                        setIsDownloading(null);
                        setDownloadProgress(0);
                        setDownloadMessage("");
                    }
                } catch (e) {
                    logger.error("Failed to parse SSE data:", e);
                }
            };

            eventSource.onerror = () => {
                eventSource.close();
                logger.error("SSE connection error");
                setIsDownloading(null);
                setDownloadProgress(0);
                setDownloadMessage("");
            };
        } catch (error) {
            logger.error("Download failed:", error);
            alert("Failed to download model. Check backend logs.");
            setIsDownloading(null);
            setDownloadProgress(0);
            setDownloadMessage("");
        }
    };

    const handleAudioReady = async (audioBlob: Blob) => {
        if (!model) {
            setTranscription("Error: Please select a model first.");
            return;
        }

        setIsProcessing(true);
        setTranscription("");
        setProcessingTime(0);

        logger.debug(`Sending audio blob to server, size: ${audioBlob.size} bytes`);

        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        formData.append("language", language);
        formData.append("use_itn", useItn.toString());
        formData.append("model", model);

        try {
            // Use SSE endpoint for streaming response with heartbeat
            const response = await fetch("http://127.0.0.1:8000/transcribe_stream", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error("Failed to get response reader");
            }

            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE messages
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || ""; // Keep incomplete message in buffer

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.status === "processing") {
                                // Heartbeat received, update processing time
                                setProcessingTime(data.heartbeat);
                                logger.debug(`[SSE] Heartbeat: ${data.heartbeat}`);
                            } else if (data.status === "complete") {
                                // Final result
                                const text = data.text;
                                if (text) {
                                    setTranscription(text);
                                } else {
                                    setTranscription("[No speech detected]");
                                }
                                logger.debug("[SSE] Transcription complete");
                            } else if (data.status === "error") {
                                setTranscription(`Error: ${data.message}`);
                                logger.error("[SSE] Transcription error:", data.message);
                            }
                        } catch (e) {
                            logger.error("[SSE] Failed to parse message:", e);
                        }
                    }
                }
            }
        } catch (error) {
            logger.error("Transcription error:", error);
            setTranscription("Error: Could not transcribe audio. Ensure backend is running.");
        } finally {
            setIsProcessing(false);
            setProcessingTime(0);
        }
    };

    const copyToClipboard = () => {
        navigator.clipboard.writeText(transcription);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="min-h-screen relative overflow-hidden">
            {/* Animated Background */}
            <div className="animated-bg">
                <div className="gradient-orb gradient-orb-1" />
                <div className="gradient-orb gradient-orb-2" />
                <div className="gradient-orb gradient-orb-3" />
            </div>

            {/* Noise Texture */}
            <div className="noise-overlay" />

            {/* Main Content */}
            <div className="relative z-10 container mx-auto max-w-2xl px-4 py-12 min-h-screen flex flex-col">

                {/* Header */}
                <motion.header
                    className="flex items-center justify-between mb-12"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                >
                    <div className="flex items-center gap-4">
                        <motion.div
                            className="p-3 rounded-2xl bg-linear-to-br from-cyan-500 to-blue-600 shadow-lg"
                            style={{ boxShadow: '0 0 40px rgba(6, 182, 212, 0.4)' }}
                            whileHover={{ scale: 1.05, rotate: 5 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            <Bot className="w-8 h-8 text-white" />
                        </motion.div>
                        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-linear-to-r from-white via-cyan-100 to-cyan-300">
                            STT AI
                        </h1>
                    </div>

                    {!showSettings && (
                        <button
                            onClick={() => setShowSettings(true)}
                            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-white/80 hover:text-white"
                            title="Settings"
                        >
                            <Settings className="w-6 h-6" />
                        </button>
                    )}
                </motion.header>

                {/* Main Card */}
                {/* Main Card */}
                <AnimatePresence mode="wait">
                    {showSettings ? (
                        <SettingsPage
                            key="settings"
                            availableModels={availableModels}
                            isDownloading={isDownloading}
                            downloadProgress={downloadProgress}
                            downloadMessage={downloadMessage}
                            onDownload={handleDownloadModel}
                            onBack={() => setShowSettings(false)}
                            isLoadingModels={isLoadingModels}
                            serverStatus={serverStatus}
                        />
                    ) : (
                        <motion.main
                            key="main"
                            className="flex-1 flex flex-col gap-6"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                        >
                            <div className="glass-card glass-card-light p-8">
                                <AudioInput
                                    onAudioReady={handleAudioReady}
                                    isProcessing={isProcessing}
                                    isStreamingMode={model === "Paraformer"}
                                    isStreaming={isStreaming}
                                    onStartStreaming={() => {
                                        setTranscription("");
                                        setIsProcessing(true);
                                        startStreaming();
                                    }}
                                    onStopStreaming={() => {
                                        stopStreaming();
                                        // Keep isProcessing=true to show loader until final result
                                        setIsProcessing(true);
                                    }}
                                />

                                {/* Settings Quick View */}
                                <motion.div
                                    className="mt-8 flex flex-col items-center gap-4"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    {/* No model selected warning */}
                                    {!model && downloadedModels.length === 0 && (
                                        <div className="text-amber-400/90 text-sm text-center px-4 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                                            No models downloaded. Please go to Settings to download a model.
                                        </div>
                                    )}
                                    {!model && downloadedModels.length > 0 && (
                                        <div className="text-cyan-400/90 text-sm text-center px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                                            Please select a model to start transcribing.
                                        </div>
                                    )}
                                    {/* Model loading indicator */}
                                    {model && modelLoadingStatus === 'loading' && (
                                        <div className="text-cyan-400/90 text-sm text-center px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center gap-2">
                                            <div className="w-4 h-4 border-2 border-cyan-400/40 border-t-cyan-400 rounded-full animate-spin" />
                                            <span>Loading model...</span>
                                        </div>
                                    )}
                                    {model && modelLoadingStatus === 'ready' && (
                                        <div className="text-emerald-400/90 text-xs text-center px-3 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                                            ✓ Model ready
                                        </div>
                                    )}

                                    <div className="flex flex-wrap justify-center gap-4">
                                        {/* Model Selector - Only downloaded models */}
                                        <select
                                            value={model || ""}
                                            onChange={(e) => {
                                                const newModel = e.target.value;
                                                handleModelChange(newModel);
                                                // Reset language logic based on model capability
                                                if (newModel === "Paraformer") {
                                                    setLanguage("zh");
                                                } else {
                                                    setLanguage("auto");
                                                }
                                            }}
                                            className="glass-select"
                                            title="Select Model"
                                            disabled={downloadedModels.length === 0}
                                        >
                                            {downloadedModels.length === 0 ? (
                                                <option value="">No models available</option>
                                            ) : (
                                                <>
                                                    {!model && <option value="">Select a model...</option>}
                                                    {downloadedModels.map((name) => (
                                                        <option key={name} value={name}>
                                                            {name}
                                                        </option>
                                                    ))}
                                                </>
                                            )}
                                        </select>

                                        <select
                                            value={language}
                                            onChange={(e) => setLanguage(e.target.value)}
                                            className="glass-select"
                                            disabled={!model || model === "Paraformer"}
                                            title={model === "Paraformer" ? "Language locked to Chinese" : "Select Language"}
                                        >
                                            {(!model || model === "SenseVoiceSmall") && (
                                                <>
                                                    <option value="auto">Auto Detect</option>
                                                    <option value="zh">Chinese</option>
                                                    <option value="en">English</option>
                                                    <option value="ja">Japanese</option>
                                                    <option value="ko">Korean</option>
                                                    <option value="yue">Cantonese</option>
                                                </>
                                            )}
                                            {model === "Paraformer" && (
                                                <option value="zh">Chinese</option>
                                            )}
                                            {model === "Fun-ASR-Nano" && (
                                                <>
                                                    <option value="auto">Auto Detect</option>
                                                    <option value="zh">Chinese</option>
                                                    <option value="en">English</option>
                                                    <option value="yue">Cantonese</option>
                                                    <option value="ja">Japanese</option>
                                                    <option value="ko">Korean</option>
                                                    {/* Other supported languages */}
                                                    <option value="ar">Arabic</option>
                                                    <option value="bg">Bulgarian</option>
                                                    <option value="hr">Croatian</option>
                                                    <option value="cs">Czech</option>
                                                    <option value="da">Danish</option>
                                                    <option value="nl">Dutch</option>
                                                    <option value="et">Estonian</option>
                                                    <option value="fil">Filipino</option>
                                                    <option value="fi">Finnish</option>
                                                    <option value="el">Greek</option>
                                                    <option value="hi">Hindi</option>
                                                    <option value="hu">Hungarian</option>
                                                    <option value="id">Indonesian</option>
                                                    <option value="ga">Irish</option>
                                                    <option value="lv">Latvian</option>
                                                    <option value="lt">Lithuanian</option>
                                                    <option value="ms">Malay</option>
                                                    <option value="mt">Maltese</option>
                                                    <option value="pl">Polish</option>
                                                    <option value="pt">Portuguese</option>
                                                    <option value="ro">Romanian</option>
                                                    <option value="sk">Slovak</option>
                                                    <option value="sl">Slovenian</option>
                                                    <option value="sv">Swedish</option>
                                                    <option value="th">Thai</option>
                                                    <option value="vi">Vietnamese</option>
                                                </>
                                            )}
                                        </select>

                                        <button
                                            onClick={() => setUseItn(!useItn)}
                                            className={cn("glass-toggle", useItn && "active")}
                                            disabled={model === "Paraformer"}
                                            title={model === "Paraformer" ? "ITN not available in streaming mode" : "Toggle Inverse Text Normalization"}
                                        >
                                            <Settings2 className="w-4 h-4" />
                                            <span>ITN {useItn ? "On" : "Off"}</span>
                                        </button>
                                    </div>
                                </motion.div>
                            </div>

                            {/* Result Area */}
                            <AnimatePresence mode="wait">
                                {(transcription || isProcessing) && (
                                    <motion.div
                                        className="result-card p-6"
                                        initial={{ opacity: 0, y: 20, scale: 0.98 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: -10, scale: 0.98 }}
                                        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                                    >
                                        <div className="flex items-center justify-between mb-4">
                                            <span className="text-xs font-medium text-white/40 uppercase tracking-wider">
                                                Transcription
                                            </span>
                                            <AnimatePresence mode="wait">
                                                {transcription && (
                                                    <motion.button
                                                        key="copy-btn"
                                                        onClick={copyToClipboard}
                                                        className="copy-button"
                                                        title="Copy text"
                                                        initial={{ opacity: 0, scale: 0.8 }}
                                                        animate={{ opacity: 1, scale: 1 }}
                                                        exit={{ opacity: 0, scale: 0.8 }}
                                                        whileHover={{ scale: 1.1 }}
                                                        whileTap={{ scale: 0.9 }}
                                                    >
                                                        <AnimatePresence mode="wait">
                                                            {copied ? (
                                                                <motion.div
                                                                    key="check"
                                                                    initial={{ opacity: 0, rotate: -90 }}
                                                                    animate={{ opacity: 1, rotate: 0 }}
                                                                    exit={{ opacity: 0, rotate: 90 }}
                                                                >
                                                                    <Check className="w-4 h-4 text-emerald-400" />
                                                                </motion.div>
                                                            ) : (
                                                                <motion.div
                                                                    key="copy"
                                                                    initial={{ opacity: 0, rotate: -90 }}
                                                                    animate={{ opacity: 1, rotate: 0 }}
                                                                    exit={{ opacity: 0, rotate: 90 }}
                                                                >
                                                                    <Copy className="w-4 h-4" />
                                                                </motion.div>
                                                            )}
                                                        </AnimatePresence>
                                                    </motion.button>
                                                )}
                                            </AnimatePresence>
                                        </div>

                                        {isProcessing && !transcription && !streamingText ? (
                                            <div className="space-y-3">
                                                <div className="flex items-center gap-2 text-white/60 text-sm">
                                                    <div className="w-4 h-4 border-2 border-cyan-400/40 border-t-cyan-400 rounded-full animate-spin" />
                                                    <span>Processing... {processingTime > 0 ? `${processingTime}s` : ''}</span>
                                                </div>
                                                <div className="skeleton h-4 w-3/4" />
                                                <div className="skeleton h-4 w-1/2" />
                                            </div>
                                        ) : (
                                            <motion.p
                                                className="text-lg leading-relaxed text-white/90 whitespace-pre-wrap"
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                                transition={{ delay: 0.1 }}
                                            >
                                                {transcription}
                                                {streamingText && (
                                                    <span className="text-cyan-400/80">{streamingText}</span>
                                                )}
                                            </motion.p>
                                        )}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.main>
                    )}
                </AnimatePresence>

                <motion.footer
                    className="mt-12 text-center text-sm text-white/20"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                >
                    Powered by FunASR
                </motion.footer>
            </div>
        </div>
    );
}

export default App;
