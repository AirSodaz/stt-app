type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const IS_DEV = import.meta.env.DEV;

class Logger {
    private formatMessage(level: LogLevel, message: string): string {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour12: false });
        return `[${timeString}] [${level.toUpperCase()}] ${message}`;
    }

    debug(message: string, ...args: any[]) {
        if (IS_DEV) {
            console.debug(this.formatMessage('debug', message), ...args);
        }
    }

    info(message: string, ...args: any[]) {
        console.info(this.formatMessage('info', message), ...args);
    }

    warn(message: string, ...args: any[]) {
        console.warn(this.formatMessage('warn', message), ...args);
    }

    error(message: string, ...args: any[]) {
        console.error(this.formatMessage('error', message), ...args);
    }
}

export const logger = new Logger();
