import { createContext, useContext } from 'react';
import { cn } from '../../lib/utils';

type TabsContextType = {
    value: string;
    onChange: (value: string) => void;
    variant: 'default' | 'segment';
};

const TabsContext = createContext<TabsContextType | undefined>(undefined);

export interface TabsProps {
    value: string;
    onChange: (value: string) => void;
    variant?: 'default' | 'segment';
    className?: string;
    children: React.ReactNode;
}

export function Tabs({ value, onChange, variant = 'default', className, children }: TabsProps) {
    return (
        <TabsContext.Provider value={{ value, onChange, variant }}>
            <div className={cn(
                "inline-flex rounded-xl p-1 gap-1 border border-transparent",
                // Common container styles
                "bg-black/5 dark:bg-white/5 border-black/5 dark:border-white/10",
                className
            )}>
                {children}
            </div>
        </TabsContext.Provider>
    );
}

export interface TabProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    value: string;
}

export function Tab({ value, className, children, ...props }: TabProps) {
    const context = useContext(TabsContext);
    if (!context) throw new Error("Tab must be used within Tabs");

    const isActive = context.value === value;

    return (
        <button
            onClick={() => context.onChange(value)}
            className={cn(
                "relative flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500",

                // Variant: Default (Cyan tint)
                context.variant === 'default' && isActive && "bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 shadow-lg shadow-cyan-500/10",
                context.variant === 'default' && !isActive && "text-slate-600 hover:text-slate-800 hover:bg-black/5 dark:text-white/70 dark:hover:text-white/90 dark:hover:bg-white/5",

                // Variant: Segment (Card-like)
                context.variant === 'segment' && isActive && "bg-white dark:bg-white/10 shadow-sm text-cyan-600 dark:text-cyan-400",
                context.variant === 'segment' && !isActive && "text-slate-600 dark:text-white/60 hover:text-slate-900 dark:hover:text-white",

                // Make segment tabs expand
                context.variant === 'segment' && "flex-1",

                className
            )}
            {...props}
        >
            {children}
        </button>
    );
}
