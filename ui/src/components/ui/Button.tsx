import { ButtonHTMLAttributes, forwardRef } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'primary' | 'secondary' | 'ghost' | 'glow' | 'glass';
    size?: 'default' | 'sm' | 'icon' | 'lg';
    asMotion?: boolean;
}

// Combine generic button props with motion props
type CombinedProps = ButtonProps & HTMLMotionProps<"button">;

export const Button = forwardRef<HTMLButtonElement, CombinedProps>(
    ({ className, variant = 'primary', size = 'default', asMotion = false, children, ...props }, ref) => {
        const baseStyles = "inline-flex items-center justify-center rounded-xl font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 disabled:opacity-50 disabled:pointer-events-none";

        const variants = {
            primary: "bg-cyan-500 text-white hover:bg-cyan-600 shadow-lg shadow-cyan-500/20",
            secondary: "bg-black/5 hover:bg-black/10 dark:bg-white/5 dark:hover:bg-white/10 text-slate-700 hover:text-slate-900 dark:text-white/80 dark:hover:text-white",
            ghost: "hover:bg-black/5 dark:hover:bg-white/5 text-slate-700 dark:text-white/80",
            glow: "glow-button text-white shadow-lg shadow-cyan-500/20",
            glass: "glass-upload" // Reusing the glass-upload style logic for generic glass buttons if needed
        };

        const sizes = {
            default: "h-10 px-4 py-2",
            sm: "h-8 px-3 text-xs",
            lg: "h-12 px-8 text-lg",
            icon: "h-10 w-10 p-2",
        };

        const combinedClassName = cn(
            baseStyles,
            variants[variant],
            sizes[size],
            className
        );

        if (asMotion) {
            return (
                <motion.button
                    ref={ref}
                    className={combinedClassName}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    {...props}
                >
                    {children}
                </motion.button>
            );
        }

        return (
            <button
                ref={ref}
                className={combinedClassName}
                {...props}
            >
                {children}
            </button>
        );
    }
);

Button.displayName = "Button";
