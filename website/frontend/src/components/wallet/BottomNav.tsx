'use client';

import { useWalletStore, type WalletTab } from '@/lib/wallet-store';
import { t } from '@/lib/translations';
import { LayoutDashboard, Send, QrCode, List, User } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

const navItems: { tab: WalletTab; labelKey: 'home' | 'sendNav' | 'receiveNav' | 'activityNav' | 'profileNav'; icon: React.ElementType }[] = [
  { tab: 'dashboard', labelKey: 'home', icon: LayoutDashboard },
  { tab: 'send', labelKey: 'sendNav', icon: Send },
  { tab: 'receive', labelKey: 'receiveNav', icon: QrCode },
  { tab: 'transactions', labelKey: 'activityNav', icon: List },
  { tab: 'profile', labelKey: 'profileNav', icon: User },
];

export function BottomNav() {
  const activeTab = useWalletStore((s) => s.activeTab);
  const setActiveTab = useWalletStore((s) => s.setActiveTab);
  const lang = useWalletStore((s) => s.language);
  const l = t[lang];

  return (
    <nav className='absolute bottom-0 left-0 right-0 bg-background/90 backdrop-blur-xl border-t border-border z-50'>
      <div className='flex items-center justify-around h-[62px]'>
        {navItems.map(({ tab, labelKey, icon: Icon }) => {
          const isActive = activeTab === tab;
          return (
            <motion.button
              key={tab} onClick={() => setActiveTab(tab)}
              className='relative flex flex-col items-center gap-0.5 px-2.5 py-1 rounded-xl min-w-[48px] min-h-[42px] justify-center'
              aria-label={l[labelKey]} aria-current={isActive ? 'page' : undefined}
              whileHover={{ y: -2, scale: 1.08 }}
              whileTap={{ scale: 0.88 }}
            >
              {isActive && (
                <motion.div layoutId='nav-pill' className='absolute inset-0 bg-secondary rounded-xl' transition={{ type: 'spring', stiffness: 400, damping: 30 }} />
              )}
              <Icon className={cn('w-[18px] h-[18px] relative z-10 transition-colors', isActive ? 'text-primary' : 'text-muted-foreground')} strokeWidth={isActive ? 2.4 : 1.6} />
              <span className={cn('text-[8px] font-medium leading-tight relative z-10 transition-colors', isActive ? 'text-primary' : 'text-muted-foreground')}>
                {l[labelKey]}
              </span>
            </motion.button>
          );
        })}
      </div>
    </nav>
  );
}