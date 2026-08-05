'use client';

import { useWalletStore } from '@/lib/wallet-store';
import { t } from '@/lib/translations';
import { formatCurrency, formatDate, getInitials, localName, type Transaction, type Bilingual, type Lang } from '@/lib/wallet-data';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { ArrowUpRight, ArrowDownLeft, TrendingUp, TrendingDown, Eye, EyeOff, Bell } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

const fu = (i: number) => ({ initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: i * 0.05, duration: 0.3, ease: 'easeOut' } });

export function DashboardScreen() {
  const { balance, transactions, setActiveTab, language, profileName } = useWalletStore();
  const [balanceVisible, setBalanceVisible] = useState(true);
  const l = t[language];
  const recentTx = transactions.slice(0, 6);
  const income = transactions.filter(tx => tx.type === 'received' && tx.status === 'completed').reduce((s, tx) => s + tx.amount, 0);
  const expense = transactions.filter(tx => tx.type === 'sent' && tx.status === 'completed').reduce((s, tx) => s + tx.amount, 0);
  const greetingName = profileName[language];

  return (
    <div className='px-4 pt-2 h-full flex flex-col'>
      <motion.div className='flex items-center justify-between mb-2.5' {...fu(0)}>
        <motion.div whileHover={{ x: 2 }} className='cursor-default'>
          <p className='text-[11px] text-muted-foreground'>{l.namaste}! {greetingName}</p>
          <h1 className='text-lg font-bold tracking-tight'>{l.appName}</h1>
        </motion.div>
        <motion.button className='relative w-9 h-9 flex items-center justify-center rounded-full bg-secondary hover:bg-accent transition-colors' whileHover={{ scale: 1.15, rotate: 15 }} whileTap={{ scale: 0.9 }}>
          <Bell className='w-4 h-4' />
          <motion.span className='absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full' animate={{ scale: [1, 1.5, 1], opacity: [1, 0.6, 1] }} transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }} />
        </motion.button>
      </motion.div>

      <motion.div {...fu(1)}>
        <motion.div whileHover={{ y: -3, scale: 1.01 }} transition={{ type: 'spring', stiffness: 300, damping: 20 }}>
          <Card className='mb-2.5 overflow-hidden'>
            <CardContent className='p-4'>
              <div className='flex items-center justify-between mb-0.5'>
                <motion.span className='text-[11px] text-muted-foreground font-medium' whileHover={{ letterSpacing: '0.05em' }}>{l.totalBalance}</motion.span>
                <motion.button onClick={() => setBalanceVisible(!balanceVisible)} className='w-7 h-7 flex items-center justify-center rounded-full hover:bg-secondary transition-colors' whileHover={{ scale: 1.2, rotate: 10 }} whileTap={{ scale: 0.85 }} aria-label={balanceVisible ? l.hideBalance : l.showBalance}>
                  {balanceVisible ? <Eye className='w-3.5 h-3.5' /> : <EyeOff className='w-3.5 h-3.5' />}
                </motion.button>
              </div>
              <h2 className='text-[26px] font-bold tracking-tight mb-3 tabular-nums'>{balanceVisible ? formatCurrency(balance, language) : '••••••'}</h2>
              <div className='flex gap-2.5'>
                <motion.button onClick={() => setActiveTab('send')} className='flex-1 flex items-center justify-center gap-1.5 h-10 bg-primary text-primary-foreground rounded-xl font-medium text-xs hover:opacity-90 transition-all' whileHover={{ scale: 1.05, y: -2, boxShadow: '0 8px 25px -5px rgba(0,0,0,0.15)' }} whileTap={{ scale: 0.95 }}>
                  <motion.div whileHover={{ x: 1, y: -1 }}><ArrowUpRight className='w-3.5 h-3.5' /></motion.div>{l.send}
                </motion.button>
                <motion.button onClick={() => setActiveTab('receive')} className='flex-1 flex items-center justify-center gap-1.5 h-10 bg-secondary text-secondary-foreground rounded-xl font-medium text-xs hover:bg-accent transition-all' whileHover={{ scale: 1.05, y: -2, boxShadow: '0 8px 25px -5px rgba(0,0,0,0.08)' }} whileTap={{ scale: 0.95 }}>
                  <motion.div whileHover={{ x: -1, y: -1 }}><ArrowDownLeft className='w-3.5 h-3.5' /></motion.div>{l.receive}
                </motion.button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      <div className='grid grid-cols-2 gap-2.5 mb-2.5'>
        <motion.div {...fu(2)}><motion.div whileHover={{ y: -4, scale: 1.03, boxShadow: '0 12px 30px -8px rgba(16,185,129,0.15)' }} transition={{ type: 'spring', stiffness: 300, damping: 20 }}><Card className='overflow-hidden'><CardContent className='p-3'><div className='flex items-center gap-1.5 mb-1'><motion.div className='w-6 h-6 rounded-md bg-emerald-100 dark:bg-emerald-950 flex items-center justify-center' whileHover={{ rotate: 10, scale: 1.15 }}><TrendingUp className='w-3 h-3 text-emerald-600 dark:text-emerald-400' /></motion.div><span className='text-[10px] text-muted-foreground font-medium'>{l.income}</span></div><p className='text-sm font-bold tabular-nums text-emerald-600 dark:text-emerald-400'>+{formatCurrency(income, language)}</p></CardContent></Card></motion.div></motion.div>
        <motion.div {...fu(3)}><motion.div whileHover={{ y: -4, scale: 1.03, boxShadow: '0 12px 30px -8px rgba(239,68,68,0.15)' }} transition={{ type: 'spring', stiffness: 300, damping: 20 }}><Card className='overflow-hidden'><CardContent className='p-3'><div className='flex items-center gap-1.5 mb-1'><motion.div className='w-6 h-6 rounded-md bg-red-100 dark:bg-red-950 flex items-center justify-center' whileHover={{ rotate: -10, scale: 1.15 }}><TrendingDown className='w-3 h-3 text-red-600 dark:text-red-400' /></motion.div><span className='text-[10px] text-muted-foreground font-medium'>{l.expense}</span></div><p className='text-sm font-bold tabular-nums text-red-600 dark:text-red-400'>-{formatCurrency(expense, language)}</p></CardContent></Card></motion.div></motion.div>
      </div>

      <motion.div className='flex-1 min-h-0 flex flex-col' {...fu(4)}>
        <div className='flex items-center justify-between mb-2'>
          <h3 className='text-sm font-semibold'>{l.recentActivity}</h3>
          <motion.button onClick={() => setActiveTab('transactions')} className='text-[11px] text-muted-foreground hover:text-foreground transition-colors' whileHover={{ x: 2, scale: 1.05 }} whileTap={{ scale: 0.92 }}>{l.seeAll}</motion.button>
        </div>
        <div className='flex-1 min-h-0 overflow-y-auto space-y-0.5 pr-0.5'>
          {recentTx.map((tx, i) => (
            <motion.div
              key={tx.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 + i * 0.05, duration: 0.25 }}
              whileHover={{ x: 4 }}
              className='rounded-xl overflow-hidden'
            >
              <TxRow transaction={tx} lang={language} />
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

function TxRow({ transaction: tx, lang }: { transaction: Transaction; lang: Lang }) {
  const isR = tx.type === 'received';
  return (
    <div className='flex items-center gap-2.5 p-2.5 rounded-xl bg-transparent hover:bg-accent/60 transition-all duration-200 ease-out cursor-default'>
      <Avatar className='w-8 h-8 shrink-0 transition-transform duration-200 ease-out group-hover:scale-105'><AvatarFallback className={cn('text-[10px] font-semibold', isR ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300' : 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300')}>{getInitials(localName(tx.name, lang))}</AvatarFallback></Avatar>
      <div className='flex-1 min-w-0'>
        <p className='text-xs font-medium truncate'>{localName(tx.name, lang)}</p>
        <p className='text-[10px] text-muted-foreground truncate'>{localName(tx.description, lang)}</p>
      </div>
      <div className='text-right shrink-0'>
        <p className={cn('text-xs font-semibold tabular-nums', isR ? 'text-emerald-600 dark:text-emerald-400' : 'text-foreground')}>{isR ? '+' : '-'}{formatCurrency(tx.amount, lang)}</p>
        <p className='text-[9px] text-muted-foreground'>{formatDate(tx.date, lang)}</p>
      </div>
    </div>
  );
}