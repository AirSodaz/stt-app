import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { AudioInput } from './components/AudioInput';
import { Bot, Copy, Check, Settings2, Settings, FileAudio, Radio, Trash2, ChevronDown, ChevronUp, Download } from 'lucide-react';
import { cn } from './lib/utils';
import { SettingsPage } from './components/SettingsPage';
import { useStreamingASR } from './hooks/useStreamingASR';
import { logger } from './lib/logger';
import { SaveOptionsModal } from './components/SaveOptionsModal';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { join } from '@tauri-apps/api/path';

function App() {
    const { t } = useTranslation();
    const [transcription, setTranscription] = useState("");
    const [isProcessing, setIsProcessing] = useState(false);
    const [copied, setCopied] = useState(false);
    const [language, setLanguage] = useState("auto");
    const [useItn, setUseItn] = useState(true);
    // Model is null until we load preference or user selects one
    const [model, setModel] = useState<string | null>(null);
    const [availableModels, setAvailableModels] = useState<{ name: string, downloaded: boolean, type?: string }[]>([]);
    const [downloadedModels, setDownloadedModels] = useState<string[]>([]);
    const [isDownloading, setIsDownloading] = useState<string | null>(null);
    const [downloadProgress, setDownloadProgress] = useState<number>(0);

    const [downloadMessage, setDownloadMessage] = useState<string>("");
    const [showSettings, setShowSettings] = useState(false);
    const [activeTab, setActiveTab] = useState<'offline' | 'realtime'>('offline');

    // Server connection state
    const [isLoadingModels, setIsLoadingModels] = useState(true);
    const [serverStatus, setServerStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');

    // Model loading state: 'idle' = no model loaded, 'loading' = loading in progress, 'ready' = model ready
    const [modelLoadingStatus, setModelLoadingStatus] = useState<'idle' | 'loading' | 'ready'>('idle');

    // Streaming state
    const [streamingText, setStreamingText] = useState("");

    // Processing time tracking (in seconds)
    const [processingTime, setProcessingTime] = useState<number>(0);

    // Batch processing state
    interface BatchItem {
        id: string;
        file: File;
        status: 'pending' | 'processing' | 'complete' | 'error';
        result?: string;
        processingTime?: number;
    }
    const [batchQueue, setBatchQueue] = useState<BatchItem[]>([]);
    const [isBatchProcessing, setIsBatchProcessing] = useState(false);
    const [expandedBatchItems, setExpandedBatchItems] = useState<Set<string>>(new Set());
    const [showSaveModal, setShowSaveModal] = useState(false);

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
        setDownloadMessage(t('settings.download') + "...");

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
            setTranscription(t('app.error.selectModel'));
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
                                    setTranscription(t('app.error.speech'));
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
            setTranscription(t('app.error.transcribe'));
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

    // Batch processing handlers
    const handleMultipleFilesReady = useCallback((files: File[]) => {
        const items: BatchItem[] = files.map((file) => ({
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            file,
            status: 'pending' as const,
        }));
        setBatchQueue(items);
        setExpandedBatchItems(new Set());
        // Auto-start processing
        processBatchQueue(items);
    }, [model, language, useItn]);

    const processBatchItem = async (item: BatchItem): Promise<BatchItem> => {
        const formData = new FormData();
        formData.append("file", item.file, item.file.name);
        formData.append("language", language);
        formData.append("use_itn", useItn.toString());
        if (model) formData.append("model", model);

        let processingTime = 0;

        try {
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
            let result = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.status === "processing") {
                                processingTime = data.heartbeat;
                            } else if (data.status === "complete") {
                                result = data.text || t('app.error.speech');
                            } else if (data.status === "error") {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            if (e instanceof SyntaxError) continue;
                            throw e;
                        }
                    }
                }
            }

            return { ...item, status: 'complete', result, processingTime };
        } catch (error) {
            logger.error(`Batch item error for ${item.file.name}:`, error);
            return { ...item, status: 'error', result: `Error: ${error}`, processingTime };
        }
    };

    const processBatchQueue = async (items: BatchItem[]) => {
        if (!model) {
            logger.error("No model selected for batch processing");
            return;
        }

        setIsBatchProcessing(true);

        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            // Update status to processing
            setBatchQueue(prev => prev.map(q =>
                q.id === item.id ? { ...q, status: 'processing' as const } : q
            ));

            const processedItem = await processBatchItem(item);

            // Update with result
            setBatchQueue(prev => prev.map(q =>
                q.id === item.id ? processedItem : q
            ));

            // Auto-expand completed item
            setExpandedBatchItems(prev => new Set([...prev, item.id]));
        }

        setIsBatchProcessing(false);
    };

    const copyAllBatchResults = () => {
        const allResults = batchQueue
            .filter(item => item.status === 'complete' && item.result)
            .map(item => `[${item.file.name}]\n${item.result}`)
            .join('\n\n');
        navigator.clipboard.writeText(allResults);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const saveBatchResults = () => {
        setShowSaveModal(true);
    };

    const handleSaveConfirm = async (mode: 'merge' | 'separate', path: string) => {
        try {
            const completedItems = batchQueue.filter(item => item.status === 'complete' && item.result);
            if (completedItems.length === 0) {
                alert("No completed items to save.");
                return;
            }

            if (mode === 'merge') {
                const allResults = completedItems
                    .map(item => `[${item.file.name}]\n${item.result}`)
                    .join('\n\n');

                const timestamp = new Date().toISOString().slice(0, 19).replace('T', '_').replace(/:/g, '-');
                const filename = `batch_transcription_${timestamp}.txt`;
                const fullPath = await join(path, filename);

                await writeTextFile(fullPath, allResults);
                alert(t('saveModal.success', { path: fullPath }));
            } else {
                let savedCount = 0;
                for (const item of completedItems) {
                    // Simple filename sanitization or just append .txt
                    const originalName = item.file.name;
                    const nameWithoutExt = originalName.lastIndexOf('.') !== -1
                        ? originalName.substring(0, originalName.lastIndexOf('.'))
                        : originalName;

                    const filename = `${nameWithoutExt}.txt`;
                    const fullPath = await join(path, filename);

                    await writeTextFile(fullPath, item.result || "");
                    savedCount++;
                }
                alert(t('saveModal.successCount', { count: savedCount, path }));
            }
        } catch (error) {
            logger.error("Failed to save files:", error);
            alert(`Failed to save: ${error}`);
        }
    };

    const clearBatchQueue = () => {
        setBatchQueue([]);
        setExpandedBatchItems(new Set());
    };

    const toggleBatchItemExpand = (id: string) => {
        setExpandedBatchItems(prev => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
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
                        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-linear-to-r from-cyan-600 via-cyan-500 to-cyan-400 dark:from-white dark:via-cyan-100 dark:to-cyan-300">
                            {t('app.title')}
                        </h1>
                    </div>

                    {!showSettings && (
                        <button
                            onClick={() => setShowSettings(true)}
                            className="p-2 rounded-xl bg-black/5 hover:bg-black/10 dark:bg-white/5 dark:hover:bg-white/10 transition-colors text-slate-700 hover:text-slate-900 dark:text-white/80 dark:hover:text-white"
                            title={t('app.settings')}
                        >
                            <Settings className="w-6 h-6" />
                        </button>
                    )}
                </motion.header>

                {/* Tabs */}
                {!showSettings && (
                    <div className="flex justify-center mb-8">
                        <div className="bg-black/5 dark:bg-white/5 p-1 rounded-xl flex gap-1 border border-black/5 dark:border-white/10">
                            <button
                                onClick={() => {
                                    setActiveTab('offline');
                                    // Reset model if current one is not valid for this tab
                                    const currentModelType = availableModels.find(m => m.name === model)?.type || 'offline';
                                    if (currentModelType === 'online') setModel(null);
                                }}
                                className={cn(
                                    "px-6 py-2 rounded-lg flex items-center gap-2 transition-all text-sm font-medium",
                                    activeTab === 'offline'
                                        ? "bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 shadow-lg shadow-cyan-500/10"
                                        : "text-slate-600 hover:text-slate-800 hover:bg-black/5 dark:text-white/70 dark:hover:text-white/90 dark:hover:bg-white/5"
                                )}
                            >
                                <FileAudio className="w-4 h-4" />
                                {t('app.offline')}
                            </button>
                            <button
                                onClick={() => {
                                    setActiveTab('realtime');
                                    // Reset model if current one is not valid for this tab
                                    const currentModelType = availableModels.find(m => m.name === model)?.type || 'offline';
                                    if (currentModelType !== 'online') setModel(null);
                                }}
                                className={cn(
                                    "px-6 py-2 rounded-lg flex items-center gap-2 transition-all text-sm font-medium",
                                    activeTab === 'realtime'
                                        ? "bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 shadow-lg shadow-cyan-500/10"
                                        : "text-slate-600 hover:text-slate-800 hover:bg-black/5 dark:text-white/70 dark:hover:text-white/90 dark:hover:bg-white/5"
                                )}
                            >
                                <Radio className="w-4 h-4" />
                                {t('app.realtime')}
                            </button>
                        </div>
                    </div>
                )}

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
                                    onMultipleFilesReady={handleMultipleFilesReady}
                                    isProcessing={isProcessing || isBatchProcessing}
                                    isStreamingMode={activeTab === 'realtime'}
                                    isStreaming={isStreaming}
                                    showUpload={activeTab === 'offline'}
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
                                        <div className="text-amber-600 dark:text-amber-400/90 text-sm text-center px-4 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                                            {t('app.model.noneDownloaded')}
                                        </div>
                                    )}
                                    {!model && downloadedModels.length > 0 && (
                                        <div className="text-cyan-600 dark:text-cyan-400/90 text-sm text-center px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                                            {t('app.model.select')}
                                        </div>
                                    )}
                                    {/* Model loading indicator */}
                                    {model && modelLoadingStatus === 'loading' && (
                                        <div className="text-cyan-600 dark:text-cyan-400/90 text-sm text-center px-4 py-2 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center gap-2">
                                            <div className="w-4 h-4 border-2 border-cyan-600/40 dark:border-cyan-400/40 border-t-cyan-600 dark:border-t-cyan-400 rounded-full animate-spin" />
                                            <span>{t('app.model.loading')}</span>
                                        </div>
                                    )}
                                    {model && modelLoadingStatus === 'ready' && (
                                        <div className="text-emerald-600 dark:text-emerald-400/90 text-xs text-center px-3 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                                            ✓ {t('app.model.ready')}
                                        </div>
                                    )}

                                    <div className="flex flex-wrap justify-center gap-4">
                                        {/* Model Selector - Filtered by Tab */}
                                        <select
                                            value={model || ""}
                                            onChange={(e) => {
                                                const newModel = e.target.value;
                                                handleModelChange(newModel);
                                                // Reset language logic based on model capability
                                                if (newModel === "Paraformer-Online" || newModel === "Paraformer") {
                                                    setLanguage("zh");
                                                } else {
                                                    setLanguage("auto");
                                                }
                                                // Auto-enable ITN for models that support it, disable for streaming (online) models
                                                const modelType = availableModels.find(m => m.name === newModel)?.type || 'offline';
                                                if (modelType === 'online') {
                                                    setUseItn(false);
                                                } else {
                                                    setUseItn(true);
                                                }
                                            }}
                                            className="glass-select"
                                            title="Select Model"
                                            disabled={downloadedModels.length === 0}
                                        >
                                            {downloadedModels.filter(name => {
                                                const m = availableModels.find(am => am.name === name);
                                                const type = m?.type || 'offline';
                                                return activeTab === 'realtime' ? type === 'online' : type !== 'online';
                                            }).length === 0 ? (
                                                <option value="">{t('app.model.noneAvailable', { mode: activeTab === 'offline' ? t('app.offline') : t('app.realtime') })}</option>
                                            ) : (
                                                <>
                                                    {!model && <option value="">{t('app.model.placeholder')}</option>}
                                                    {downloadedModels
                                                        .filter(name => {
                                                            const m = availableModels.find(am => am.name === name);
                                                            const type = m?.type || 'offline';
                                                            return activeTab === 'realtime' ? type === 'online' : type !== 'online';
                                                        })
                                                        .map((name) => (
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
                                            title={model === "Paraformer-Online" || model === "Paraformer" ? t('app.model.locked') : t('app.model.selectLanguage')}
                                        >
                                            {(!model || model === "SenseVoiceSmall") && (
                                                <>
                                                    <option value="auto">{t('app.language.auto')}</option>
                                                    <option value="zh">{t('app.language.zh')}</option>
                                                    <option value="en">{t('app.language.en')}</option>
                                                    <option value="ja">{t('app.language.ja')}</option>
                                                    <option value="ko">{t('app.language.ko')}</option>
                                                    <option value="yue">{t('app.language.yue')}</option>
                                                </>
                                            )}
                                            {(model === "Paraformer" || model === "Paraformer-Online") && (
                                                <option value="zh">{t('app.language.zh')}</option>
                                            )}
                                            {model === "Fun-ASR-Nano" && (
                                                <>
                                                    <option value="auto">{t('app.language.auto')}</option>
                                                    <option value="zh">{t('app.language.zh')}</option>
                                                    <option value="en">{t('app.language.en')}</option>
                                                    <option value="yue">{t('app.language.yue')}</option>
                                                    <option value="ja">{t('app.language.ja')}</option>
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
                                            disabled={model === "Paraformer-Online"}
                                            title={model === "Paraformer-Online" ? "ITN not available in streaming mode" : "Toggle Inverse Text Normalization"}
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
                                            <span className="text-xs font-medium text-secondary uppercase tracking-wider">
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
                                                <div className="flex items-center gap-2 text-secondary text-sm">
                                                    <div className="w-4 h-4 border-2 border-cyan-400/40 border-t-cyan-400 rounded-full animate-spin" />
                                                    <span>Processing... {processingTime > 0 ? `${processingTime}s` : ''}</span>
                                                </div>
                                                <div className="skeleton h-4 w-3/4" />
                                                <div className="skeleton h-4 w-1/2" />
                                            </div>
                                        ) : (
                                            <motion.p
                                                className="text-lg leading-relaxed text-primary whitespace-pre-wrap"
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

                            {/* Batch Processing Results */}
                            <AnimatePresence>
                                {batchQueue.length > 0 && (
                                    <motion.div
                                        className="result-card p-6"
                                        initial={{ opacity: 0, y: 20, scale: 0.98 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: -10, scale: 0.98 }}
                                        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                                    >
                                        {/* Header */}
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="flex items-center gap-3">
                                                <span className="text-xs font-medium text-secondary uppercase tracking-wider">
                                                    Batch Processing
                                                </span>
                                                <span className="text-xs text-secondary">
                                                    {batchQueue.filter(i => i.status === 'complete').length}/{batchQueue.length} completed
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {batchQueue.some(i => i.status === 'complete') && (
                                                    <>
                                                        <motion.button
                                                            onClick={saveBatchResults}
                                                            className="copy-button flex items-center gap-1 text-xs mr-2"
                                                            title="Save as .txt"
                                                            whileHover={{ scale: 1.05 }}
                                                            whileTap={{ scale: 0.95 }}
                                                        >
                                                            <Download className="w-3 h-3" />
                                                            <span>Save</span>
                                                        </motion.button>
                                                        <motion.button
                                                            onClick={copyAllBatchResults}
                                                            className="copy-button flex items-center gap-1 text-xs"
                                                            title="Copy all results"
                                                            whileHover={{ scale: 1.05 }}
                                                            whileTap={{ scale: 0.95 }}
                                                        >
                                                            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                                                            <span>Copy All</span>
                                                        </motion.button>
                                                    </>
                                                )}
                                                {!isBatchProcessing && (
                                                    <motion.button
                                                        onClick={clearBatchQueue}
                                                        className="p-1.5 rounded-lg bg-white/5 hover:bg-red-500/20 text-white/40 hover:text-red-400 transition-colors"
                                                        title="Clear queue"
                                                        whileHover={{ scale: 1.05 }}
                                                        whileTap={{ scale: 0.95 }}
                                                    >
                                                        <Trash2 className="w-4 h-4" />
                                                    </motion.button>
                                                )}
                                            </div>
                                        </div>

                                        {/* Progress Bar */}
                                        {isBatchProcessing && (
                                            <div className="mb-4">
                                                <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                                                    <motion.div
                                                        className="h-full bg-linear-to-r from-cyan-500 to-blue-500"
                                                        initial={{ width: 0 }}
                                                        animate={{
                                                            width: `${(batchQueue.filter(i => i.status === 'complete' || i.status === 'error').length / batchQueue.length) * 100}%`
                                                        }}
                                                        transition={{ duration: 0.3 }}
                                                    />
                                                </div>
                                            </div>
                                        )}

                                        {/* File List */}
                                        <div className="space-y-2 max-h-96 overflow-y-auto">
                                            {batchQueue.map((item) => (
                                                <motion.div
                                                    key={item.id}
                                                    className={cn(
                                                        "rounded-lg border transition-colors",
                                                        item.status === 'processing' && "bg-cyan-500/10 border-cyan-500/30",
                                                        item.status === 'complete' && "bg-emerald-500/5 border-emerald-500/20",
                                                        item.status === 'error' && "bg-red-500/10 border-red-500/30",
                                                        item.status === 'pending' && "bg-white/5 border-white/10"
                                                    )}
                                                    layout
                                                >
                                                    {/* Item Header */}
                                                    <button
                                                        onClick={() => toggleBatchItemExpand(item.id)}
                                                        className="w-full px-4 py-3 flex items-center justify-between text-left"
                                                        disabled={item.status === 'pending'}
                                                    >
                                                        <div className="flex items-center gap-3 min-w-0">
                                                            {/* Status Icon */}
                                                            {item.status === 'pending' && (
                                                                <div className="w-4 h-4 rounded-full bg-white/20" />
                                                            )}
                                                            {item.status === 'processing' && (
                                                                <div className="w-4 h-4 border-2 border-cyan-400/40 border-t-cyan-400 rounded-full animate-spin" />
                                                            )}
                                                            {item.status === 'complete' && (
                                                                <Check className="w-4 h-4 text-emerald-400" />
                                                            )}
                                                            {item.status === 'error' && (
                                                                <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-white text-xs">!</div>
                                                            )}

                                                            {/* Filename */}
                                                            <span className={cn(
                                                                "text-sm truncate",
                                                                item.status === 'complete' && "text-primary",
                                                                item.status === 'processing' && "text-cyan-700 dark:text-cyan-400",
                                                                item.status === 'error' && "text-red-700 dark:text-red-400",
                                                                item.status === 'pending' && "text-secondary"
                                                            )}>
                                                                {item.file.name}
                                                            </span>

                                                            {/* Processing Time */}
                                                            {item.processingTime && item.processingTime > 0 && (
                                                                <span className="text-xs text-secondary">
                                                                    {item.processingTime}s
                                                                </span>
                                                            )}
                                                        </div>

                                                        {/* Expand Icon */}
                                                        {(item.status === 'complete' || item.status === 'error') && (
                                                            expandedBatchItems.has(item.id)
                                                                ? <ChevronUp className="w-4 h-4 text-secondary" />
                                                                : <ChevronDown className="w-4 h-4 text-secondary" />
                                                        )}
                                                    </button>

                                                    {/* Expanded Content */}
                                                    <AnimatePresence>
                                                        {expandedBatchItems.has(item.id) && item.result && (
                                                            <motion.div
                                                                initial={{ height: 0, opacity: 0 }}
                                                                animate={{ height: 'auto', opacity: 1 }}
                                                                exit={{ height: 0, opacity: 0 }}
                                                                transition={{ duration: 0.2 }}
                                                                className="overflow-hidden"
                                                            >
                                                                <div className="px-4 pb-3 pt-0">
                                                                    <p className={cn(
                                                                        "text-sm leading-relaxed whitespace-pre-wrap",
                                                                        item.status === 'error' ? "text-red-700 dark:text-red-300" : "text-secondary"
                                                                    )}>
                                                                        {item.result}
                                                                    </p>
                                                                </div>
                                                            </motion.div>
                                                        )}
                                                    </AnimatePresence>
                                                </motion.div>
                                            ))}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.main>
                    )}
                </AnimatePresence>

                <motion.footer
                    className="mt-12 text-center text-sm text-secondary"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.5 }}
                    transition={{ delay: 0.8 }}
                >
                    Powered by FunASR
                </motion.footer>
            </div>

            <SaveOptionsModal
                isOpen={showSaveModal}
                onClose={() => setShowSaveModal(false)}
                onConfirm={handleSaveConfirm}
            />
        </div >
    );
}

export default App;
