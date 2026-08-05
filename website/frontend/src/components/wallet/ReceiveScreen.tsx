'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useWalletStore } from '@/lib/wallet-store';
import { t } from '@/lib/translations';
import { Logo } from './Logo';
import { Copy, Share2, Check } from 'lucide-react';
import { useState, useMemo } from 'react';
import { useToast } from '@/hooks/use-toast';
import { motion } from 'framer-motion';

export function ReceiveScreen() {
  const { profileName, language } = useWalletStore();
  const { toast } = useToast();
  const l = t[language];
  const [copied, setCopied] = useState(false);
  const accountId = `${profileName.en.toLowerCase().replace(/\s+/g, '.')}@neuralsentinel.com`;
  const qrCells = useMemo(() => Array.from({ length: 49 }, (_, i) => ((i * 7 + 42) * 13) % 10 > 4), []);

  const handleCopy = async () => {
    try { await navigator.clipboard.writeText(accountId); setCopied(true); toast({ title: l.copied, description: l.copiedDesc }); setTimeout(() => setCopied(false), 2000); } catch { toast({ title: 'Error', variant: 'destructive' }); }
  };

  return (
    <div className='px-4 pt-2 h-full flex flex-col'>
      <motion.h1 className='text-lg font-bold tracking-tight mb-0.5' initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2 }}>{l.receiveMoney}</motion.h1>
      <motion.p className='text-[11px] text-muted-foreground mb-2.5' initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.08 }}>{l.shareWalletIdHint}</motion.p>

      <motion.div className='flex-1 flex flex-col items-center justify-start' initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.12, duration: 0.3, type: 'spring' }}>
        <motion.div whileHover={{ y: -4, scale: 1.02 }} transition={{ type: 'spring', stiffness: 300 }}>
          <Card className='w-full mb-2.5'><CardContent className='p-4 flex flex-col items-center'>
            <div className='w-32 h-32 bg-white dark:bg-zinc-900 rounded-2xl p-2 mb-3 shadow-sm border border-border/50'>
              <div className='w-full h-full rounded-lg flex flex-col items-center justify-center gap-1.5'>
                <Logo size={28} animate={false} />
                <div className='grid grid-cols-7 gap-[1.5px]'>
                  {qrCells.map((filled, i) => <div key={i} className={`w-[5px] h-[5px] rounded-[0.5px] ${filled ? 'bg-foreground' : 'bg-foreground/10'}`} />)}
                </div>
              </div>
            </div>
            <h3 className='font-semibold text-sm'>{profileName[language]}</h3>
            <p className='text-[11px] text-muted-foreground mt-0.5 font-mono'>{accountId}</p>
          </CardContent></Card>
        </motion.div>

        <div className='grid grid-cols-2 gap-2.5 w-full mb-2.5'>
          <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}><Button variant='outline' className='w-full h-10 rounded-xl gap-1.5 text-xs' onClick={handleCopy}>{copied ? <Check className='w-3.5 h-3.5' /> : <Copy className='w-3.5 h-3.5' />}{copied ? l.copied : l.copyId}</Button></motion.div>
          <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}><Button variant='outline' className='w-full h-10 rounded-xl gap-1.5 text-xs' onClick={() => {}}><Share2 className='w-3.5 h-3.5' />{l.share}</Button></motion.div>
        </div>

        <Card className='w-full'><CardContent className='p-3.5'>
          <h4 className='text-xs font-semibold mb-2'>{l.howItWorks}</h4>
          <ol className='space-y-1.5 text-[11px] text-muted-foreground'>
            {[l.step1, l.step2, l.step3].map((text, i) => (
              <motion.li key={i} className='flex gap-2' initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.35 + i * 0.06 }} whileHover={{ x: 2 }}>
                <span className='w-4 h-4 rounded-full bg-primary text-primary-foreground text-[9px] flex items-center justify-center shrink-0 font-bold'>{i + 1}</span>{text}
              </motion.li>
            ))}
          </ol>
        </CardContent></Card>
      </motion.div>
    </div>
  );
}