import { forwardRef } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';

export interface CardProps extends HTMLMotionProps<"div"> {
    variant?: 'glass' | 'glass-light' | 'result';
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, variant = 'glass', children, ...props }, ref) => {
        return (
            <motion.div
                ref={ref}
                className={cn(
                    variant === 'glass' && "glass-card",
                    variant === 'glass-light' && "glass-card glass-card-light",
                    variant === 'result' && "result-card",
                    className
                )}
                {...props}
            >
                {children}
            </motion.div>
        );
    }
);

Card.displayName = "Card";
