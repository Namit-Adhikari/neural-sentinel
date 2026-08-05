'use client';

import { useWalletStore, type WalletTab } from '@/lib/wallet-store';
import { DashboardScreen } from './DashboardScreen';
import { SendScreen } from './SendScreen';
import { ReceiveScreen } from './ReceiveScreen';
import { TransactionsScreen } from './TransactionsScreen';
import { ProfileScreen } from './ProfileScreen';
import { BottomNav } from './BottomNav';
import { AnimatePresence, motion } from 'framer-motion';

const screenMap: Record<WalletTab, React.ComponentType> = {
  dashboard: DashboardScreen, send: SendScreen, receive: ReceiveScreen,
  transactions: TransactionsScreen, profile: ProfileScreen,
};

export function WalletContainer() {
  const activeTab = useWalletStore((s) => s.activeTab);
  const ActiveScreen = screenMap[activeTab];
  return (
    <div className='h-full flex flex-col bg-background relative'>
      <main className='flex-1 overflow-hidden pb-[62px]'>
        <AnimatePresence mode='wait'>
          <motion.div key={activeTab} initial={{ opacity: 0, y: 10, filter: 'blur(2px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }} exit={{ opacity: 0, y: -8, filter: 'blur(2px)' }} transition={{ duration: 0.2, ease: 'easeOut' }} className='h-full'>
            <ActiveScreen />
          </motion.div>
        </AnimatePresence>
      </main>
      <BottomNav />
    </div>
  );
}
