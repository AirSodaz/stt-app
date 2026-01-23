import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

const resources = {
    en: {
        translation: {
            "app": {
                "title": "STT AI",
                "settings": "Settings",
                "copy": "Copy text",
                "copied": "Copied",
                "offline": "Offline",
                "realtime": "Real-time",
                "error": {
                    "selectModel": "Error: Please select a model first.",
                    "transcribe": "Error: Could not transcribe audio. Ensure backend is running.",
                    "speech": "[No speech detected]",
                    "microphone": "Could not access microphone."
                },
                "model": {
                    "noneDownloaded": "No models downloaded. Please go to Settings to download a model.",
                    "select": "Please select a model to start transcribing.",
                    "loading": "Loading model...",
                    "ready": "Model ready",
                    "noneAvailable": "No {{mode}} models available",
                    "placeholder": "Select a model...",
                    "locked": "Language locked to Chinese",
                    "selectLanguage": "Select Language"
                },
                "language": {
                    "auto": "Auto Detect",
                    "zh": "Chinese",
                    "en": "English",
                    "ja": "Japanese",
                    "ko": "Korean",
                    "yue": "Cantonese"
                }
            },
            "audioInput": {
                "drop": "Drop audio files here",
                "recording": "Recording...",
                "processing": "Processing...",
                "tapToSpeak": "Tap to Speak",
                "tapToStop": "Tap again to stop",
                "dragDropHint": "Or drag & drop audio files here",
                "upload": "Upload Audio"
            },
            "batch": {
                "title": "Batch Processing",
                "dropHint": "Drop audio files here to add to queue",
                "processingQueue": "Processing queue...",
                "complete": "Batch Complete",
                "copyAll": "Copy All Results",
                "clear": "Clear Queue",
                "save": "Save All",
                "status": {
                    "pending": "Pending",
                    "processing": "Processing",
                    "complete": "Complete",
                    "error": "Error"
                }
            },
            "settings": {
                "title": "Settings",
                "back": "Go back",
                "appearance": "Appearance",
                "theme": {
                    "light": "Light",
                    "dark": "Dark",
                    "auto": "Auto"
                },
                "language": "Language",
                "modelManagement": "Model Management",
                "waitingBackend": "Waiting for backend...",
                "connectionError": "Failed to connect to backend server.",
                "ensureRunning": "Please ensure the server is running.",
                "downloaded": "Downloaded",
                "download": "Download"
            },
            "saveModal": {
                "title": "Save Results",
                "selectFolder": "Select Destination Folder",
                "mode": "Save Mode",
                "singleFile": "Single File",
                "separateFiles": "Separate Files",
                "destination": "Destination",
                "placeholder": "Select a folder...",
                "browse": "Browse",
                "cancel": "Cancel",
                "confirm": "Save Results",
                "success": "Successfully saved to {{path}}",
                "successCount": "Successfully saved {{count}} files to {{path}}"
            }
        }
    },
    zh: {
        translation: {
            "app": {
                "title": "STT AI",
                "settings": "设置",
                "copy": "复制文本",
                "copied": "已复制",
                "offline": "离线文件",
                "realtime": "实时语音",
                "error": {
                    "selectModel": "错误：请先选择一个模型。",
                    "transcribe": "错误：无法转录音频。请确保后台正在运行。",
                    "speech": "[未检测到语音]",
                    "microphone": "无法访问麦克风。"
                },
                "model": {
                    "noneDownloaded": "未下载模型。请前往设置页面下载模型。",
                    "select": "请选择一个模型以开始转录。",
                    "loading": "正在加载模型...",
                    "ready": "模型就绪",
                    "noneAvailable": "无可用{{mode}}模型",
                    "placeholder": "选择模型...",
                    "locked": "语言已锁定为中文",
                    "selectLanguage": "选择语言"
                },
                "language": {
                    "auto": "自动检测",
                    "zh": "中文",
                    "en": "英语",
                    "ja": "日语",
                    "ko": "韩语",
                    "yue": "粤语"
                }
            },
            "audioInput": {
                "drop": "将音频文件拖放到此处",
                "recording": "正在录音...",
                "processing": "处理中...",
                "tapToSpeak": "点击说话",
                "tapToStop": "再次点击停止",
                "dragDropHint": "或拖放音频文件到此处",
                "upload": "上传音频"
            },
            "batch": {
                "title": "批量处理",
                "dropHint": "拖放音频文件到此处以添加到队列",
                "processingQueue": "正在处理队列...",
                "complete": "批量处理完成",
                "copyAll": "复制所有结果",
                "clear": "清空队列",
                "save": "保存所有",
                "status": {
                    "pending": "等待中",
                    "processing": "处理中",
                    "complete": "完成",
                    "error": "错误"
                }
            },
            "settings": {
                "title": "设置",
                "back": "返回",
                "appearance": "外观",
                "theme": {
                    "light": "浅色",
                    "dark": "深色",
                    "auto": "自动"
                },
                "language": "语言",
                "modelManagement": "模型管理",
                "waitingBackend": "等待后台...",
                "connectionError": "无法连接到后台服务器。",
                "ensureRunning": "请确保服务器正在运行。",
                "downloaded": "已下载",
                "download": "下载"
            },
            "saveModal": {
                "title": "保存结果",
                "selectFolder": "选择目标文件夹",
                "mode": "保存模式",
                "singleFile": "单个文件",
                "separateFiles": "分离文件",
                "destination": "目标位置",
                "placeholder": "选择文件夹...",
                "browse": "浏览",
                "cancel": "取消",
                "confirm": "保存结果",
                "success": "成功保存到 {{path}}",
                "successCount": "成功保存 {{count}} 个文件到 {{path}}"
            }
        }
    }
};

i18n
    .use(LanguageDetector)
    .use(initReactI18next)
    .init({
        resources,
        fallbackLng: 'en',
        interpolation: {
            escapeValue: false
        }
    });

export default i18n;
