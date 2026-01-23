import { motion } from 'framer-motion';
import { cn } from '../../lib/utils';

export interface SegmentedControlOption<T extends string> {
    value: T;
    label: React.ReactNode;
    icon?: React.ReactNode;
}

interface SegmentedControlProps<T extends string> {
    options: SegmentedControlOption<T>[];
    value: T;
    onChange: (value: T) => void;
    className?: string;
}

export function SegmentedControl<T extends string>({
    options,
    value,
    onChange,
    className
}: SegmentedControlProps<T>) {
    return (
        <div className={cn("flex gap-2 p-1 bg-black/5 dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/10", className)}>
            {options.map((option) => {
                const isSelected = value === option.value;
                return (
                    <button
                        key={option.value}
                        onClick={() => onChange(option.value)}
                        className={cn(
                            "relative flex-1 flex items-center justify-center gap-2 py-2 rounded-lg transition-all text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500",
                            isSelected
                                ? "text-cyan-600 dark:text-cyan-400"
                                : "text-slate-600 hover:text-slate-800 hover:bg-black/5 dark:text-white/70 dark:hover:text-white/90 dark:hover:bg-white/5"
                        )}
                    >
                        {isSelected && (
                            <motion.div
                                layoutId="segmented-bg"
                                className="absolute inset-0 bg-cyan-500/20 shadow-lg shadow-cyan-500/10 rounded-lg"
                                initial={false}
                                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                            />
                        )}
                        <span className="relative z-10 flex items-center gap-2">
                            {option.icon}
                            {option.label}
                        </span>
                    </button>
                );
            })}
        </div>
    );
}
