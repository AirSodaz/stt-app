import { motion, AnimatePresence } from 'framer-motion';
import { X, FolderOpen, FileText, Files } from 'lucide-react';
import { useState } from 'react';
import { open } from '@tauri-apps/plugin-dialog';

interface SaveOptionsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (mode: 'merge' | 'separate', path: string) => void;
}

export const SaveOptionsModal = ({ isOpen, onClose, onConfirm }: SaveOptionsModalProps) => {
    const [mode, setMode] = useState<'merge' | 'separate'>('merge');
    const [path, setPath] = useState<string>('');

    const handleBrowse = async () => {
        try {
            const selected = await open({
                directory: true,
                multiple: false,
                title: 'Select Destination Folder'
            });

            if (selected && typeof selected === 'string') {
                setPath(selected);
            }
        } catch (error) {
            console.error('Failed to open dialog:', error);
        }
    };

    const handleConfirm = () => {
        if (!path) return;
        onConfirm(mode, path);
        onClose();
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        className="fixed inset-0 bg-black/60 backdrop-blur-xs z-50"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                    />

                    {/* Modal */}
                    <div className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none p-4">
                        <motion.div
                            className="w-full max-w-md bg-[#0f1115] border border-white/10 rounded-2xl shadow-2xl overflow-hidden pointer-events-auto"
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
                                <h3 className="text-lg font-medium text-white">Save Results</h3>
                                <button
                                    onClick={onClose}
                                    className="p-1 rounded-lg hover:bg-white/5 text-white/40 hover:text-white transition-colors"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="p-6 space-y-6">
                                {/* Mode Selection */}
                                <div className="space-y-3">
                                    <label className="text-sm font-medium text-white/60 uppercase tracking-wider">
                                        Save Mode
                                    </label>
                                    <div className="grid grid-cols-2 gap-3">
                                        <button
                                            onClick={() => setMode('merge')}
                                            className={`p-4 rounded-xl border flex flex-col items-center gap-3 transition-colors ${mode === 'merge'
                                                ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
                                                : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10 hover:border-white/20'
                                                }`}
                                        >
                                            <FileText className="w-6 h-6" />
                                            <span className="text-sm font-medium">Single File</span>
                                        </button>
                                        <button
                                            onClick={() => setMode('separate')}
                                            className={`p-4 rounded-xl border flex flex-col items-center gap-3 transition-colors ${mode === 'separate'
                                                ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
                                                : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10 hover:border-white/20'
                                                }`}
                                        >
                                            <Files className="w-6 h-6" />
                                            <span className="text-sm font-medium">Separate Files</span>
                                        </button>
                                    </div>
                                </div>

                                {/* Path Selection */}
                                <div className="space-y-3">
                                    <label className="text-sm font-medium text-white/60 uppercase tracking-wider">
                                        Destination
                                    </label>
                                    <div className="flex gap-2">
                                        <div className="flex-1 px-4 py-2.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white/50 truncate">
                                            {path || 'Select a folder...'}
                                        </div>
                                        <button
                                            onClick={handleBrowse}
                                            className="px-4 py-2 bg-white/10 hover:bg-white/15 border border-white/10 rounded-lg text-white transition-colors flex items-center gap-2"
                                        >
                                            <FolderOpen className="w-4 h-4" />
                                            <span>Browse</span>
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Footer */}
                            <div className="p-6 bg-white/5 border-t border-white/5 flex justify-end gap-3">
                                <button
                                    onClick={onClose}
                                    className="px-4 py-2 text-sm font-medium text-white/60 hover:text-white transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleConfirm}
                                    disabled={!path}
                                    className="px-6 py-2 bg-linear-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white text-sm font-medium rounded-lg shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                >
                                    Save Results
                                </button>
                            </div>
                        </motion.div>
                    </div>
                </>
            )}
        </AnimatePresence>
    );
};
