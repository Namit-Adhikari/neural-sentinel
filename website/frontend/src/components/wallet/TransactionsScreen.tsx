'use client';

import { useWalletStore } from '@/lib/wallet-store';
import { t } from '@/lib/translations';
import { formatDate, getInitials, formatCurrency, localName, type Transaction, type Lang } from '@/lib/wallet-data';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Search, ArrowDownLeft, ArrowUpRight, Clock, CheckCircle2, XCircle } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { motion } from 'framer-motion';

const sc = {
  completed: { labelKey: 'done' as const, icon: CheckCircle2, cls: 'text-emerald-600 dark:text-emerald-400' },
  pending: { labelKey: 'pending' as const, icon: Clock, cls: 'text-amber-600 dark:text-amber-400' },
  failed: { labelKey: 'failed' as const, icon: XCircle, cls: 'text-red-600 dark:text-red-400' },
};

const filterKeys = ['all', 'sent', 'received', 'pending'] as const;

export function TransactionsScreen() {
  const { transactions, txFilter, setTxFilter, txSearchQuery, setTxSearchQuery, language } = useWalletStore();
  const l = t[language];

  const filtered = transactions.filter(tx => {
    const mf = txFilter === 'all' || tx.type === txFilter;
    const ms = tx.name.en.toLowerCase().includes(txSearchQuery.toLowerCase()) || tx.name.ne.includes(txSearchQuery) || localName(tx.description, language).includes(txSearchQuery);
    return mf && ms;
  });

  const grouped: { key: string; date: string; txs: Transaction[] }[] = [];
  let gi = 0, cd = '';
  for (const tx of filtered) {
    const dl = formatDate(tx.date, language);
    if (dl !== cd) { cd = dl; gi++; grouped.push({ key: `g${gi}-${dl}`, date: dl, txs: [] }); }
    grouped[gi - 1].txs.push(tx);
  }

  return (
    <div className='px-4 pt-2 h-full flex flex-col'>
      <motion.h1 className='text-lg font-bold tracking-tight mb-2' initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2 }}>{l.activity}</motion.h1>

      <motion.div className='relative mb-2' initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06 }}>
        <Search className='absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground' />
        <Input value={txSearchQuery} onChange={(e) => setTxSearchQuery(e.target.value)} placeholder={l.searchTransactions} className='pl-9 h-9 rounded-xl text-xs' />
      </motion.div>

      <motion.div className='flex gap-1.5 mb-2' initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        {filterKeys.map(f => (
          <motion.button key={f} onClick={() => setTxFilter(f)} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className={cn('px-3 py-1.5 rounded-full text-[11px] font-medium transition-colors', txFilter === f ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground hover:text-foreground')}>
            {l[f]}
          </motion.button>
        ))}
      </motion.div>

      <div className='flex-1 min-h-0 overflow-y-auto pr-0.5'>
        {grouped.length === 0 ? <div className='text-center py-8'><p className='text-xs text-muted-foreground'>{l.noTransactions}</p></div> : (
          <div className='space-y-2.5'>{grouped.map((g, gi) => (
            <motion.div key={g.key} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 + gi * 0.05 }}>
              <p className='text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1'>{g.date}</p>
              <Card><CardContent className='p-1.5'>{g.txs.map(tx => <TxRow key={tx.id} tx={tx} lang={language} />)}</CardContent></Card>
            </motion.div>
          ))}</div>
        )}
      </div>
    </div>
  );
}

function TxRow({ tx, lang }: { tx: Transaction; lang: Lang }) {
  const isR = tx.type === 'received'; const isP = tx.type === 'pending'; const s = sc[tx.status]; const SI = s.icon;
  return (
    <motion.div className='flex items-center gap-2.5 p-2 rounded-xl transition-colors cursor-default' whileHover={{ x: 3, backgroundColor: 'var(--accent)' }}>
      <Avatar className='w-7 h-7 shrink-0'><AvatarFallback className={cn('text-[9px] font-semibold', isR ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300' : isP ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300' : 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300')}>{getInitials(localName(tx.name, lang))}</AvatarFallback></Avatar>
      <div className='flex-1 min-w-0'>
        <div className='flex items-center gap-1'><p className='text-[11px] font-medium truncate'>{localName(tx.name, lang)}</p><SI className={cn('w-3 h-3 shrink-0', s.cls)} /></div>
        <p className='text-[10px] text-muted-foreground truncate'>{localName(tx.description, lang)}</p>
      </div>
      <div className='text-right shrink-0'>
        <div className='flex items-center justify-end gap-0.5'>
          {isR ? <ArrowDownLeft className='w-3 h-3 text-emerald-600 dark:text-emerald-400' /> : isP ? <Clock className='w-3 h-3 text-amber-600 dark:text-amber-400' /> : <ArrowUpRight className='w-3 h-3 text-muted-foreground' />}
          <p className={cn('text-[11px] font-semibold tabular-nums', isR ? 'text-emerald-600 dark:text-emerald-400' : isP ? 'text-amber-600 dark:text-amber-400' : 'text-foreground')}>{isR ? '+' : isP ? '' : '-'}{formatCurrency(tx.amount, lang)}</p>
        </div>
        <Badge variant='secondary' className='text-[8px] px-1 py-0 h-3 mt-0.5'>{t[lang][s.labelKey]}</Badge>
      </div>
    </motion.div>
  );
}