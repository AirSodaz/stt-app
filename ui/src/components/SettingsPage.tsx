import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Moon, Sun, Laptop, Languages } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../lib/utils';
import { useTheme } from '../contexts/ThemeContext';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { Tabs, Tab } from './ui/Tabs';
import { ProgressBar } from './ui/ProgressBar';

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
    const { theme, setTheme } = useTheme();
    const { t, i18n } = useTranslation();
    const backButtonRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        // Focus the back button when settings page opens for keyboard accessibility
        backButtonRef.current?.focus();
    }, []);

    // Helper to determine current language group (e.g. 'zh-CN' -> 'zh')
    const currentLang = i18n.language.split('-')[0];

    return (
        <motion.div
            className="flex-1 flex flex-col gap-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
            <div className="flex items-center gap-4 mb-2">
                <Button
                    ref={backButtonRef}
                    onClick={onBack}
                    variant="ghost"
                    size="icon"
                    aria-label={t('settings.back')}
                >
                    <ArrowLeft className="w-5 h-5" />
                </Button>
                <h2 className="text-xl font-semibold text-primary">{t('settings.title')}</h2>
            </div>

            {/* Language Section */}
            <Card variant="glass-light" className="p-6">
                <h3 className="text-sm font-medium text-secondary uppercase tracking-wider mb-4">
                    {t('settings.language')}
                </h3>
                <Tabs
                    variant="segment"
                    value={currentLang}
                    onChange={(val) => i18n.changeLanguage(val)}
                    className="w-full"
                >
                    <Tab value="zh">
                        <Languages className="w-4 h-4" />
                        中文
                    </Tab>
                    <Tab value="en">
                        <Languages className="w-4 h-4" />
                        English
                    </Tab>
                </Tabs>
            </Card>

            {/* Theme Section */}
            <Card variant="glass-light" className="p-6">
                <h3 className="text-sm font-medium text-secondary uppercase tracking-wider mb-4">
                    {t('settings.appearance')}
                </h3>
                <Tabs
                    variant="segment"
                    value={theme}
                    onChange={(val) => setTheme(val as 'light' | 'dark' | 'auto')}
                    className="w-full"
                >
                    <Tab value="light">
                        <Sun className="w-4 h-4" />
                        {t('settings.theme.light')}
                    </Tab>
                    <Tab value="dark">
                        <Moon className="w-4 h-4" />
                        {t('settings.theme.dark')}
                    </Tab>
                    <Tab value="auto">
                        <Laptop className="w-4 h-4" />
                        {t('settings.theme.auto')}
                    </Tab>
                </Tabs>
            </Card>

            <Card variant="glass-light" className="p-8">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium text-secondary uppercase tracking-wider">
                        {t('settings.modelManagement')}
                    </h3>
                    {serverStatus === 'connecting' && (
                        <span className="text-xs text-amber-500 dark:text-amber-400 flex items-center gap-1.5 animate-pulse">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400" />
                            {t('settings.waitingBackend')}
                        </span>
                    )}
                </div>

                <div className="flex flex-col gap-3">
                    {isLoadingModels ? (
                        // Loading Skeleton
                        Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="flex flex-col gap-2 px-4 py-3 rounded-xl bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                                <div className="flex items-center justify-between">
                                    <div className="h-4 w-32 bg-black/10 dark:bg-white/10 rounded animate-pulse" />
                                    <div className="h-6 w-20 bg-black/10 dark:bg-white/10 rounded-lg animate-pulse" />
                                </div>
                            </div>
                        ))
                    ) : availableModels.length === 0 && serverStatus === 'error' ? (
                        <div className="px-4 py-8 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
                            <p className="text-sm text-red-600 dark:text-red-200 mb-2">{t('settings.connectionError')}</p>
                            <p className="text-xs text-red-400 dark:text-red-200/60">{t('settings.ensureRunning')}</p>
                        </div>
                    ) : (
                        availableModels.map((m) => (
                            <div key={m.name} className="flex flex-col gap-2 px-4 py-3 rounded-xl bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/10">
                                <div className="flex items-center justify-between">
                                    <span className={cn("text-sm font-medium", m.downloaded ? "text-primary" : "text-secondary")}>
                                        {m.name}
                                    </span>
                                    {m.downloaded ? (
                                        <span className="text-emerald-500 dark:text-emerald-400 text-xs flex items-center gap-1">
                                            <span>✓</span> {t('settings.downloaded')}
                                        </span>
                                    ) : isDownloading === m.name ? (
                                        <span className="text-cyan-600 dark:text-cyan-400 text-xs">{downloadProgress}%</span>
                                    ) : (
                                        <Button
                                            variant="primary"
                                            size="sm"
                                            onClick={() => onDownload(m.name)}
                                            disabled={isDownloading !== null}
                                            className="text-xs h-7 px-3 bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-700 dark:bg-cyan-600/60 dark:hover:bg-cyan-500/80 dark:text-white"
                                        >
                                            {t('settings.download')}
                                        </Button>
                                    )}
                                </div>
                                {/* Progress bar */}
                                {isDownloading === m.name && (
                                    <div className="flex flex-col gap-1.5">
                                        <ProgressBar progress={downloadProgress} height="sm" />
                                        <span className="text-xs text-secondary text-center">{downloadMessage}</span>
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>
            </Card>
        </motion.div >
    );
}
