import { useState, useRef, useCallback } from 'react';
import { logger } from '../lib/logger';

/**
 * Hook for streaming ASR via WebSocket.
 * Sends raw PCM Int16 audio at 16kHz.
 */
export function useStreamingASR(onPartialResult: (text: string, isFinal: boolean) => void) {
    const [isStreaming, setIsStreaming] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const processorRef = useRef<ScriptProcessorNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    const startStreaming = useCallback(async () => {
        // Set streaming immediately so button shows stop state
        setIsStreaming(true);
        logger.debug('[Streaming] Starting...');

        try {
            // 1. Get microphone stream FIRST (requires user gesture)
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });
            streamRef.current = stream;
            logger.debug('[Streaming] Got microphone access');

            // 2. Set up AudioContext (also requires user gesture context)
            const audioContext = new AudioContext({ sampleRate: 16000 });
            audioContextRef.current = audioContext;

            const source = audioContext.createMediaStreamSource(stream);
            sourceRef.current = source;

            // Use ScriptProcessor to get raw PCM data
            // Buffer size: 4096 samples = 256ms at 16kHz
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            logger.debug('[Streaming] AudioContext ready');

            // 3. Connect WebSocket
            const ws = new WebSocket('ws://127.0.0.1:8000/ws/transcribe');
            wsRef.current = ws;

            ws.onopen = () => {
                logger.debug('[Streaming] WebSocket connected, starting audio processing');

                // Start audio processing only after WS is open
                processor.onaudioprocess = (e) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);

                        // Convert Float32 to Int16
                        const int16Data = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            const s = Math.max(-1, Math.min(1, inputData[i]));
                            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        }

                        ws.send(int16Data.buffer);
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Too spammy for regular debug, keeping distinct logic or just debug
                    if (data.text) {
                        logger.debug('[Streaming] Received partial:', data.text);
                        onPartialResult(data.text, data.is_final || false);
                    }

                    if (data.is_final) {
                        logger.debug('[Streaming] Received final result, closing connection');
                        // Ensure we notify the parent that it's final, even if text was empty/processed above
                        // If text was present, we already called it. If not, we must call it now.
                        if (!data.text) {
                            onPartialResult("", true);
                        }

                        ws.close();
                        wsRef.current = null;
                        setIsStreaming(false);
                    }

                    if (data.error) {
                        logger.error('[Streaming] Server error:', data.error);
                        ws.close();
                        wsRef.current = null;
                        setIsStreaming(false);
                    }
                } catch (e) {
                    logger.error('[Streaming] Failed to parse message:', e);
                }
            };

            ws.onerror = (error) => {
                logger.error('[Streaming] WebSocket error:', error);
            };

            ws.onclose = () => {
                logger.debug('[Streaming] WebSocket closed');
                // Don't set isStreaming false here - let stopStreaming handle it
            };

        } catch (err) {
            logger.error('[Streaming] Failed to start:', err);
            setIsStreaming(false);
        }
    }, [onPartialResult]);

    const stopStreaming = useCallback(() => {
        logger.debug('[Streaming] Stopping...');

        // Disconnect audio processor
        if (processorRef.current) {
            processorRef.current.disconnect();
            processorRef.current.onaudioprocess = null;
            processorRef.current = null;
        }

        // Disconnect source
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }

        // Close audio context
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }

        // Stop media stream
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // Send EOF signal - don't close WebSocket yet, let server send final result
        // Server will close connection after sending final result, or we close after timeout
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            logger.debug('[Streaming] Sending EOF signal');
            wsRef.current.send(new ArrayBuffer(0));

            // Set a timeout to close if server doesn't respond
            setTimeout(() => {
                if (wsRef.current) {
                    logger.warn('[Streaming] Timeout, closing WebSocket');
                    wsRef.current.close();
                    wsRef.current = null;
                }
            }, 10000); // 10 second timeout for transcription
        } else {
            wsRef.current = null;
        }

        // Immediately update state to indicate recording stopped
        setIsStreaming(false);
        logger.debug('[Streaming] Audio stopped, waiting for final result...');
    }, []);

    return { isStreaming, startStreaming, stopStreaming };
}
