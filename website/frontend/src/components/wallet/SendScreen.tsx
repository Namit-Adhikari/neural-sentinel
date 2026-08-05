'use client';

import { useWalletStore } from '@/lib/wallet-store';
import { t } from '@/lib/translations';
import { getInitials, formatCurrency, currencySymbol, localName, type Contact, type Lang } from '@/lib/wallet-data';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Search, Star, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { motion, AnimatePresence } from 'framer-motion';

export function SendScreen() {
  const { contacts, sendAmount, setSendAmount, selectedContact, setSelectedContact, searchQuery, setSearchQuery, balance, isSending, sendMoney, language } = useWalletStore();
  const { toast } = useToast();
  const l = t[language];
  const sym = currencySymbol(language);

  const filtered = contacts.filter(c => c.name.en.toLowerCase().includes(searchQuery.toLowerCase()) || c.name.ne.includes(searchQuery) || (c.email && c.email.toLowerCase().includes(searchQuery.toLowerCase())));
  const favs = filtered.filter(c => c.isFavorite);
  const others = filtered.filter(c => !c.isFavorite);

  const handleSend = () => {
    if (!selectedContact) { toast({ title: l.selectContact }); return; }
    const amt = parseFloat(sendAmount);
    if (!amt || amt <= 0) { toast({ title: l.invalidAmount }); return; }
    if (amt > balance) { toast({ title: l.insufficientBalance, variant: 'destructive' }); return; }
    sendMoney(selectedContact, amt, language);
    toast({ title: l.moneySent, description: `${sym} ${amt.toLocaleString('en-US')} ${language === 'en' ? `sent to ${selectedContact.name.en}` : `${selectedContact.name.ne} लाई पठाइयो`}` });
  };

  const toggleContact = (c: Contact) => setSelectedContact(selectedContact?.id === c.id ? null : c);

  return (
    <div className='px-4 pt-2 h-full flex flex-col'>
      <motion.h1 className='text-lg font-bold tracking-tight mb-2.5' initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2 }}>{l.sendMoney}</motion.h1>

      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.06, duration: 0.25 }}>
        <Card className='mb-2.5'><CardContent className='p-3.5'>
          <label className='text-[11px] text-muted-foreground font-medium mb-1 block'>{l.amount}</label>
          <div className='flex items-baseline gap-1 mb-1'>
            <span className='text-xl font-light text-muted-foreground'>{sym}</span>
            <input type='number' value={sendAmount} onChange={(e) => setSendAmount(e.target.value)} placeholder='0' className='flex-1 text-2xl font-bold bg-transparent outline-none tabular-nums placeholder:text-muted-foreground/40' min='0' step='1' />
                   </div>
          <p className='text-[10px] text-muted-foreground mb-2'>{l.available}: <span className='font-medium text-foreground'>{formatCurrency(balance, language)}</span></p>
          <AnimatePresence>{selectedContact && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className='flex items-center gap-2.5 p-2.5 bg-secondary/50 rounded-xl mb-2'>
              <Avatar className='w-7 h-7'><AvatarFallback className='text-[9px] font-semibold bg-primary text-primary-foreground'>{getInitials(localName(selectedContact.name, language))}</AvatarFallback></Avatar>
              <div className='flex-1 min-w-0'><p className='text-xs font-medium truncate'>{localName(selectedContact.name, language)}</p><p className='text-[10px] text-muted-foreground truncate'>{selectedContact.phone}</p></div>
              <Check className='w-4 h-4 text-emerald-600 dark:text-emerald-400' />
            </motion.div>
          )}</AnimatePresence>
          <motion.button onClick={handleSend} disabled={isSending || !selectedContact || !sendAmount || parseFloat(sendAmount) <= 0} className='w-full h-10 bg-primary text-primary-foreground rounded-xl font-semibold text-xs flex items-center justify-center disabled:opacity-50 transition-all' whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.97 }}>
            {isSending ? <motion.div className='w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full' animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 0.8, ease: 'linear' }} /> : l.send}
          </motion.button>
        </CardContent></Card>
      </motion.div>

      <motion.div className='relative mb-2' initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12, duration: 0.2 }}>
        <Search className='absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground' />
        <Input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder={l.searchContacts} className='pl-9 h-9 rounded-xl text-xs' />
      </motion.div>

      <div className='flex-1 min-h-0 overflow-y-auto space-y-2 pr-0.5'>
        {favs.length > 0 && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.18 }}>
          <div className='flex items-center gap-1 mb-1'><Star className='w-3 h-3 text-amber-500 fill-amber-500' /><span className='text-[10px] font-semibold text-muted-foreground uppercase tracking-wider'>{l.favorites}</span></div>
          <div className='space-y-0.5'>{favs.map((c, i) => <motion.div key={c.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.03 }}><ContactItem contact={c} selected={selectedContact?.id === c.id} onSelect={() => toggleContact(c)} lang={language} /></motion.div>)}</div>
        </motion.div>}
        {others.length > 0 && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <span className='text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1 block'>{l.allContacts}</span>
          <div className='space-y-0.5'>{others.map((c, i) => <motion.div key={c.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.32 + i * 0.03 }}><ContactItem contact={c} selected={selectedContact?.id === c.id} onSelect={() => toggleContact(c)} lang={language} /></motion.div>)}</div>
        </motion.div>}
        {filtered.length === 0 && <div className='text-center py-6'><p className='text-xs text-muted-foreground'>{l.noContactsFound}</p></div>}
      </div>
    </div>
  );
}

function ContactItem({ contact, selected, onSelect, lang }: { contact: Contact; selected: boolean; onSelect: () => void; lang: Lang }) {
  return (
    <motion.button onClick={onSelect} className={cn('w-full flex items-center gap-2.5 p-2 rounded-xl transition-colors text-left', selected ? 'bg-primary/10 ring-1 ring-primary/30' : 'hover:bg-secondary/50')} whileHover={{ x: 2 }} whileTap={{ scale: 0.98 }}>
      <Avatar className='w-8 h-8 shrink-0'><AvatarFallback className={cn('text-[9px] font-semibold', selected ? 'bg-primary text-primary-foreground' : 'bg-secondary')}>{getInitials(localName(contact.name, lang))}</AvatarFallback></Avatar>
      <div className='flex-1 min-w-0'><p className='text-xs font-medium truncate'>{localName(contact.name, lang)}</p><p className='text-[10px] text-muted-foreground truncate'>{contact.phone}</p></div>
      {selected && <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 400 }}><Check className='w-4 h-4 text-primary shrink-0' /></motion.div>}
    </motion.button>
  );
}