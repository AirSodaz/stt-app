import { HTMLAttributes, forwardRef } from 'react';
import { cn } from '../../lib/utils';

interface BadgeProps extends HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'success' | 'warning' | 'info' | 'error';
}

export const Badge = forwardRef<HTMLDivElement, BadgeProps>(
    ({ className, variant = 'default', children, ...props }, ref) => {
        const variants = {
            default: "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300",
            success: "text-emerald-600 dark:text-emerald-400/90 bg-emerald-500/10 border border-emerald-500/20",
            warning: "text-amber-600 dark:text-amber-400/90 bg-amber-500/10 border border-amber-500/20",
            info: "text-cyan-600 dark:text-cyan-400/90 bg-cyan-500/10 border border-cyan-500/20",
            error: "text-red-600 dark:text-red-400/90 bg-red-500/10 border border-red-500/20",
        };

        return (
            <div
                ref={ref}
                className={cn(
                    "inline-flex items-center justify-center rounded-lg px-3 py-1 text-xs font-medium",
                    variants[variant],
                    className
                )}
                {...props}
            >
                {children}
            </div>
        );
    }
);

Badge.displayName = "Badge";
