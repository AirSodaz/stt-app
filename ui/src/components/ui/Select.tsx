import { SelectHTMLAttributes, forwardRef } from 'react';
import { cn } from '../../lib/utils';

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {

}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
    ({ className, children, ...props }, ref) => {
        return (
            <select
                ref={ref}
                className={cn("glass-select focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500", className)}
                {...props}
            >
                {children}
            </select>
        );
    }
);

Select.displayName = "Select";
