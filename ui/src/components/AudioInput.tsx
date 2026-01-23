import { useState, useRef, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Square, Upload, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';
import { logger } from '../lib/logger';

interface AudioInputProps {
    onAudioReady: (file: Blob) => void;
    onMultipleFilesReady?: (files: File[]) => void;
    isProcessing: boolean;
    // Streaming mode props
    isStreamingMode?: boolean;
    isStreaming?: boolean;
    onStartStreaming?: () => void;
    onStopStreaming?: () => void;
    showUpload?: boolean;
    isModelReady?: boolean;
}

export const AudioInput = memo(function AudioInput({
    onAudioReady,
    onMultipleFilesReady,
    isProcessing,
    isStreamingMode = false,
    isStreaming = false,
    onStartStreaming,
    onStopStreaming,
    showUpload = true,
    isModelReady = true
}: AudioInputProps) {
    const { t } = useTranslation();
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const startRecording = async () => {
        logger.debug('[AudioInput] startRecording called, isStreamingMode:', isStreamingMode);
        // If streaming mode, use the streaming callbacks
        if (isStreamingMode && onStartStreaming) {
            logger.debug('[AudioInput] Using streaming mode');
            onStartStreaming();
            return;
        }

        logger.debug('[AudioInput] Using normal recording mode');
        // Normal recording mode
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                logger.debug(`[DEBUG] Audio blob created, size: ${blob.size} bytes`);
                onAudioReady(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            logger.error("Error accessing microphone:", err);
            alert("Could not access microphone.");
        }
    };

    const stopRecording = () => {
        // If streaming mode, use the streaming callbacks
        if (isStreamingMode && onStopStreaming) {
            onStopStreaming();
            return;
        }

        // Normal recording mode
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    // Determine if currently "active" (recording or streaming)
    const isActive = isStreamingMode ? isStreaming : isRecording;

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        if (files.length > 1 && onMultipleFilesReady) {
            // Batch mode: multiple files selected
            onMultipleFilesReady(Array.from(files));
        } else if (files.length === 1) {
            // Single file mode
            onAudioReady(files[0]);
        }
        // Reset input to allow re-selecting same files
        e.target.value = '';
    };

    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (!showUpload || isProcessing || isActive) return;
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (!showUpload || isProcessing || isActive) return;

        const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('audio/'));

        if (files.length === 0) return;

        if (files.length > 1 && onMultipleFilesReady) {
            onMultipleFilesReady(files);
        } else if (files.length === 1) {
            onAudioReady(files[0]);
        }
    };

    return (
        <div
            className={cn(
                "flex flex-col items-center gap-8 w-full p-8 rounded-3xl transition-all duration-300",
                isDragging && "bg-cyan-500/10 ring-2 ring-cyan-500/50 scale-105"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Drag Overlay Message */}
            <AnimatePresence>
                {isDragging && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="absolute inset-0 z-50 flex items-center justify-center rounded-3xl backdrop-blur-sm"
                        style={{ backgroundColor: 'var(--glass-bg)' }}
                    >
                        <div className="flex flex-col items-center gap-4 text-cyan-400">
                            <Upload className="w-16 h-16 animate-bounce" />
                            <p className="text-2xl font-bold">{t('audioInput.drop')}</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Record Button */}
            <div className="relative">
                <motion.button
                    onClick={isActive ? stopRecording : startRecording}
                    disabled={(isProcessing && !isActive) || !isModelReady}
                    aria-label={isActive ? t('audioInput.tapToStop') : t('audioInput.tapToSpeak')}
                    className={cn(
                        "relative flex items-center justify-center w-28 h-28 rounded-full transition-colors duration-300",
                        "border-2 shadow-2xl focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-cyan-500/50",
                        isActive
                            ? "bg-linear-to-br from-red-900/80 to-red-950/80 border-red-500/50"
                            : "bg-linear-to-br from-slate-900/80 to-slate-950/80 border-white/10 hover:border-cyan-500/50",
                        ((isProcessing && !isActive) || !isModelReady) && "opacity-70 cursor-not-allowed"
                    )}
                    whileHover={{ scale: (isProcessing && !isActive) || !isModelReady ? 1 : 1.05 }}
                    whileTap={{ scale: (isProcessing && !isActive) || !isModelReady ? 1 : 0.95 }}
                >
                    <AnimatePresence mode="wait">
                        {isActive ? (
                            <motion.div
                                key="stop"
                                initial={{ opacity: 0, scale: 0.5 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.5 }}
                                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                            >
                                <Square className="w-10 h-10 text-red-400 fill-current" />
                            </motion.div>
                        ) : isProcessing ? (
                            <motion.div
                                key="loader"
                                initial={{ opacity: 0, rotate: -180 }}
                                animate={{ opacity: 1, rotate: 0 }}
                                exit={{ opacity: 0, rotate: 180 }}
                                transition={{ duration: 0.3 }}
                            >
                                <Loader2 className="w-12 h-12 text-cyan-400 animate-spin" />
                            </motion.div>
                        ) : (
                            <motion.div
                                key="mic"
                                initial={{ opacity: 0, scale: 0.5 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.5 }}
                                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                            >
                                <Mic className="w-12 h-12 text-cyan-400" />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </motion.button>
            </div>

            {/* Status Text */}
            <div className="text-center space-y-2">
                <AnimatePresence mode="wait">
                    <motion.p
                        key={isActive ? "recording" : isProcessing ? "processing" : "idle"}
                        className="text-xl font-medium text-primary"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                    >
                        {isActive ? (
                            <span className="text-red-500 dark:text-red-400">{t('audioInput.recording')}</span>
                        ) : isProcessing ? (
                            <span className="text-cyan-600 dark:text-cyan-400">{t('audioInput.processing')}</span>
                        ) : (
                            t('audioInput.tapToSpeak')
                        )}
                    </motion.p>

                </AnimatePresence>
                <motion.p
                    className="text-sm text-secondary"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    {isActive ? t('audioInput.tapToStop') : t('audioInput.dragDropHint')}
                </motion.p>
            </div>

            {/* Upload Button */}
            <AnimatePresence>
                {!isActive && !isProcessing && showUpload && (
                    <>
                        <motion.button
                            className="glass-upload cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500"
                            aria-label={t('audioInput.upload')}
                            onClick={handleUploadClick}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 10 }}
                            transition={{ duration: 0.3, delay: 0.1 }}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            <Upload className="w-4 h-4" />
                            <span>{t('audioInput.upload')}</span>
                        </motion.button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="audio/*"
                            multiple
                            className="hidden"
                            onChange={handleFileUpload}
                        />
                    </>
                )}
            </AnimatePresence>
        </div >
    );
});
