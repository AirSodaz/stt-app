import { forwardRef } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends HTMLMotionProps<"button"> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'glass' | 'outline';
    size?: 'sm' | 'md' | 'lg' | 'icon';
    isLoading?: boolean;
    children?: React.ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'secondary', size = 'md', isLoading, children, disabled, ...props }, ref) => {
        return (
            <motion.button
                ref={ref}
                disabled={disabled || isLoading}
                className={cn(
                    // Base styles
                    "inline-flex items-center justify-center rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 disabled:opacity-50 disabled:pointer-events-none cursor-pointer",

                    // Variants
                    variant === 'primary' && "bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 shadow-lg shadow-cyan-500/10 hover:bg-cyan-500/30",
                    variant === 'secondary' && "bg-black/5 dark:bg-white/5 text-slate-600 dark:text-white/70 hover:text-slate-900 dark:hover:text-white hover:bg-black/10 dark:hover:bg-white/10",
                    variant === 'ghost' && "bg-transparent hover:bg-black/5 dark:hover:bg-white/5 text-slate-700 dark:text-white/80",
                    variant === 'danger' && "bg-red-500/10 text-red-600 dark:text-red-400 hover:bg-red-500/20",
                    variant === 'glass' && "glass-toggle",
                    variant === 'outline' && "border border-slate-200 dark:border-white/10 bg-transparent hover:bg-black/5 dark:hover:bg-white/5",

                    // Sizes
                    // For glass variant, we might want to respect its intrinsic padding, but standardizing is better
                    size === 'sm' && "h-8 px-3 text-xs",
                    size === 'md' && "h-10 px-4 py-2",
                    size === 'lg' && "h-12 px-8",
                    size === 'icon' && "h-10 w-10 p-2",

                    className
                )}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                {...props}
            >
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {children as React.ReactNode}
            </motion.button>
        );
    }
);

Button.displayName = "Button";
