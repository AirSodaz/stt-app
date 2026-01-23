import { HTMLAttributes, forwardRef } from 'react';
import { cn } from '../../lib/utils';

export interface LoaderProps extends HTMLAttributes<HTMLDivElement> {
    size?: 'sm' | 'md' | 'lg' | 'xl';
    variant?: 'primary' | 'secondary' | 'neutral';
}

export const Loader = forwardRef<HTMLDivElement, LoaderProps>(
    ({ className, size = 'md', variant = 'primary', ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={cn(
                    "rounded-full animate-spin border-2",

                    // Variants
                    variant === 'primary' && "border-cyan-500/30 border-t-cyan-500 dark:border-cyan-400/30 dark:border-t-cyan-400",
                    variant === 'neutral' && "border-white/30 border-t-white",

                    // Sizes
                    size === 'sm' && "w-4 h-4",
                    size === 'md' && "w-6 h-6",
                    size === 'lg' && "w-8 h-8",
                    size === 'xl' && "w-12 h-12",

                    className
                )}
                {...props}
            />
        );
    }
);

Loader.displayName = "Loader";
