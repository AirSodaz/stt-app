import { forwardRef } from 'react';
import { motion } from 'framer-motion';
import { cn } from '../../lib/utils';

export interface ProgressBarProps extends React.HTMLAttributes<HTMLDivElement> {
    progress: number; // 0 to 100
    height?: 'sm' | 'md';
}

export const ProgressBar = forwardRef<HTMLDivElement, ProgressBarProps>(
    ({ className, progress, height = 'md', ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={cn(
                    "w-full bg-black/10 dark:bg-white/10 rounded-full overflow-hidden",
                    height === 'sm' && "h-1",
                    height === 'md' && "h-2",
                    className
                )}
                {...props}
            >
                <motion.div
                    className="h-full bg-linear-to-r from-cyan-500 to-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                    transition={{ duration: 0.3, ease: "easeOut" }}
                />
            </div>
        );
    }
);

ProgressBar.displayName = "ProgressBar";
