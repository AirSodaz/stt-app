import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Square, Upload, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';

interface AudioInputProps {
    onAudioReady: (file: Blob) => void;
    isProcessing: boolean;
    // Streaming mode props
    isStreamingMode?: boolean;
    isStreaming?: boolean;
    onStartStreaming?: () => void;
    onStopStreaming?: () => void;
}

export function AudioInput({
    onAudioReady,
    isProcessing,
    isStreamingMode = false,
    isStreaming = false,
    onStartStreaming,
    onStopStreaming
}: AudioInputProps) {
    const [isRecording, setIsRecording] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const startRecording = async () => {
        console.log('[AudioInput] startRecording called, isStreamingMode:', isStreamingMode);
        // If streaming mode, use the streaming callbacks
        if (isStreamingMode && onStartStreaming) {
            console.log('[AudioInput] Using streaming mode');
            onStartStreaming();
            return;
        }

        console.log('[AudioInput] Using normal recording mode');
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
                console.log(`[DEBUG] Audio blob created, size: ${blob.size} bytes`);
                onAudioReady(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error accessing microphone:", err);
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

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onAudioReady(e.target.files[0]);
        }
    };

    return (
        <div className="flex flex-col items-center gap-8 w-full">
            {/* Main Record Button */}
            <div className="relative">
                <motion.button
                    onClick={isActive ? stopRecording : startRecording}
                    disabled={isProcessing && !isActive}
                    className={cn(
                        "relative flex items-center justify-center w-28 h-28 rounded-full transition-colors duration-300",
                        "border-2 shadow-2xl",
                        isActive
                            ? "bg-linear-to-br from-red-900/80 to-red-950/80 border-red-500/50"
                            : "bg-linear-to-br from-slate-900/80 to-slate-950/80 border-white/10 hover:border-cyan-500/50",
                        isProcessing && !isActive && "opacity-70 cursor-not-allowed"
                    )}
                    whileHover={{ scale: isProcessing && !isActive ? 1 : 1.05 }}
                    whileTap={{ scale: isProcessing && !isActive ? 1 : 0.95 }}
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
                        className="text-xl font-medium text-white/90"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                    >
                        {isActive ? (
                            <span className="text-red-400">Recording...</span>
                        ) : isProcessing ? (
                            <span className="text-cyan-400">Processing...</span>
                        ) : (
                            "Tap to Speak"
                        )}
                    </motion.p>
                </AnimatePresence>
                <motion.p
                    className="text-sm text-white/40"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    {isActive ? "Tap again to stop" : "Or upload an audio file"}
                </motion.p>
            </div>

            {/* Upload Button */}
            <AnimatePresence>
                {!isActive && !isProcessing && (
                    <motion.label
                        className="glass-upload cursor-pointer"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.3, delay: 0.1 }}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        <Upload className="w-4 h-4" />
                        <span>Upload Audio</span>
                        <input
                            type="file"
                            accept="audio/*"
                            className="hidden"
                            onChange={handleFileUpload}
                        />
                    </motion.label>
                )}
            </AnimatePresence>
        </div>
    );
}
