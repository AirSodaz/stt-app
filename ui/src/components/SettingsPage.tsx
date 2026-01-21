import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import { cn } from '../lib/utils';

interface Model {
    name: string;
    downloaded: boolean;
}

export interface SettingsPageProps {
    availableModels: Model[];
    isDownloading: string | null;
    downloadProgress: number;
    downloadMessage: string;
    onDownload: (modelName: string) => void;
    onBack: () => void;
    isLoadingModels: boolean;
    serverStatus: 'connecting' | 'connected' | 'error';
}

export function SettingsPage({
    availableModels,
    isDownloading,
    downloadProgress,
    downloadMessage,
    onDownload,
    onBack,
    isLoadingModels,
    serverStatus
}: SettingsPageProps) {
    return (
        <motion.div
            className="flex-1 flex flex-col gap-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
            <div className="flex items-center gap-4 mb-2">
                <button
                    onClick={onBack}
                    className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-white/80 hover:text-white"
                >
                    <ArrowLeft className="w-5 h-5" />
                </button>
                <h2 className="text-xl font-semibold text-white/90">Settings</h2>
            </div>

            <div className="glass-card glass-card-light p-8">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium text-white/40 uppercase tracking-wider">
                        Model Management
                    </h3>
                    {serverStatus === 'connecting' && (
                        <span className="text-xs text-amber-400 flex items-center gap-1.5 animate-pulse">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                            Waiting for backend...
                        </span>
                    )}
                </div>

                <div className="flex flex-col gap-3">
                    {isLoadingModels ? (
                        // Loading Skeleton
                        Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="flex flex-col gap-2 px-4 py-3 rounded-xl bg-white/5 border border-white/5">
                                <div className="flex items-center justify-between">
                                    <div className="h-4 w-32 bg-white/10 rounded animate-pulse" />
                                    <div className="h-6 w-20 bg-white/10 rounded-lg animate-pulse" />
                                </div>
                            </div>
                        ))
                    ) : availableModels.length === 0 && serverStatus === 'error' ? (
                        <div className="px-4 py-8 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
                            <p className="text-sm text-red-200 mb-2">Failed to connect to backend server.</p>
                            <p className="text-xs text-red-200/60">Please ensure the server is running.</p>
                        </div>
                    ) : (
                        availableModels.map((m) => (
                            <div key={m.name} className="flex flex-col gap-2 px-4 py-3 rounded-xl bg-white/5 border border-white/10">
                                <div className="flex items-center justify-between">
                                    <span className={cn("text-sm font-medium", m.downloaded ? "text-white/90" : "text-white/60")}>
                                        {m.name}
                                    </span>
                                    {m.downloaded ? (
                                        <span className="text-emerald-400 text-xs flex items-center gap-1">
                                            <span>✓</span> Downloaded
                                        </span>
                                    ) : isDownloading === m.name ? (
                                        <span className="text-cyan-400 text-xs">{downloadProgress}%</span>
                                    ) : (
                                        <button
                                            onClick={() => onDownload(m.name)}
                                            disabled={isDownloading !== null}
                                            className="text-xs px-3 py-1 rounded-lg bg-cyan-600/60 hover:bg-cyan-500/80 text-white transition-colors disabled:opacity-50"
                                        >
                                            Download
                                        </button>
                                    )}
                                </div>
                                {/* Progress bar */}
                                {isDownloading === m.name && (
                                    <div className="flex flex-col gap-1.5">
                                        <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                                            <motion.div
                                                className="h-full bg-linear-to-r from-cyan-500 to-blue-500 rounded-full"
                                                initial={{ width: 0 }}
                                                animate={{ width: `${downloadProgress}%` }}
                                                transition={{ duration: 0.3, ease: "easeOut" }}
                                            />
                                        </div>
                                        <span className="text-xs text-white/40 text-center">{downloadMessage}</span>
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </motion.div>
    );
}
