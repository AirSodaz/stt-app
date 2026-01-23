import { ButtonHTMLAttributes, forwardRef } from 'react';
import { cn } from '../../lib/utils';
import { motion, HTMLMotionProps } from 'framer-motion';

interface ToggleProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    pressed: boolean;
    onPressedChange: (pressed: boolean) => void;
}

export const Toggle = forwardRef<HTMLButtonElement, ToggleProps & HTMLMotionProps<"button">>(
    ({ className, pressed, onPressedChange, children, ...props }, ref) => {
        return (
            <motion.button
                ref={ref}
                type="button"
                onClick={() => onPressedChange(!pressed)}
                className={cn(
                    "glass-toggle focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500",
                    pressed && "active",
                    className
                )}
                aria-pressed={pressed}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                {...props}
            >
                {children}
            </motion.button>
        );
    }
);

Toggle.displayName = "Toggle";
