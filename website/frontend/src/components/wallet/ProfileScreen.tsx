'use client';

import { useWalletStore } from '@/lib/wallet-store';
import { t, type Lang } from '@/lib/translations';
import { getInitials, localName } from '@/lib/wallet-data';
import { Card, CardContent } from '@/components/ui/card';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { User, Mail, Phone, Shield, Bell, Fingerprint, ChevronRight, LogOut, HelpCircle, Moon, Sun, Languages, Palette } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { motion } from 'framer-motion';
import { useTheme } from 'next-themes';

const fu = (i: number) => ({ initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { delay: i * 0.05, duration: 0.25 } });

export function ProfileScreen() {
  const { profileName, profileEmail, profilePhone, notificationsEnabled, biometricEnabled, toggleNotifications, toggleBiometric, setIsLoggedIn, language, setLanguage } = useWalletStore();
  const { toast } = useToast();
  const { theme, setTheme } = useTheme();
  const l = t[language];

  const handleLogout = () => { toast({ title: l.loggedOut, description: l.loggedOutDesc }); setTimeout(() => setIsLoggedIn(false), 400); };

  return (
    <div className='px-4 pt-2 h-full flex flex-col overflow-y-auto'>
      <motion.h1 className='text-lg font-bold tracking-tight mb-2.5' {...fu(0)}>{l.profile}</motion.h1>

      <motion.div {...fu(1)} whileHover={{ y: -2 }} transition={{ type: 'spring', stiffness: 300 }}><Card className='mb-2.5'><CardContent className='p-3.5'>
        <div className='flex items-center gap-3'>
          <motion.div whileHover={{ scale: 1.1, rotate: 6 }} whileTap={{ scale: 0.95 }}><Avatar className='w-11 h-11'><AvatarFallback className='text-sm font-semibold bg-primary text-primary-foreground'>{getInitials(localName(profileName, language))}</AvatarFallback></Avatar></motion.div>
          <div><h2 className='text-sm font-semibold'>{localName(profileName, language)}</h2><p className='text-[10px] text-muted-foreground'>{l.personalAccount}</p></div>
        </div>
      </CardContent></Card></motion.div>

      <motion.div {...fu(2)} whileHover={{ y: -1 }} transition={{ type: 'spring', stiffness: 300 }}><Card className='mb-2.5'><CardContent className='p-3.5'>
        <h3 className='text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-3'>{l.personalInfo}</h3>
        <div className='space-y-3'>
          <div className='space-y-1'><Label className='text-[10px] text-muted-foreground'>{l.fullTitle}</Label><div className='relative'><User className='absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground' /><Input defaultValue={localName(profileName, language)} className='pl-8 h-9 rounded-xl text-xs' readOnly /></div></div>
          <div className='space-y-1'><Label className='text-[10px] text-muted-foreground'>{l.emailAddress}</Label><div className='relative'><Mail className='absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground' /><Input defaultValue={profileEmail} className='pl-8 h-9 rounded-xl text-xs' readOnly /></div></div>
          <div className='space-y-1'><Label className='text-[10px] text-muted-foreground'>{l.phoneNumber}</Label><div className='relative'><Phone className='absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground' /><Input defaultValue={profilePhone} className='pl-8 h-9 rounded-xl text-xs' readOnly /></div></div>
        </div>
      </CardContent></Card></motion.div>

      <motion.div {...fu(3)} whileHover={{ y: -1 }} transition={{ type: 'spring', stiffness: 300 }}><Card className='mb-2.5'><CardContent className='p-3.5'>
        <h3 className='text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2'>{l.settings}</h3>
        <div className='space-y-0.5'>
          <SettingRow icon={Bell} label={l.pushNotifications} toggle toggleOn={notificationsEnabled} onToggle={toggleNotifications} />
          <SettingRow icon={Fingerprint} label={l.biometricLogin} toggle toggleOn={biometricEnabled} onToggle={toggleBiometric} />
          <SettingRow icon={Languages} label={l.language} extra={
            <div className='flex bg-secondary rounded-lg p-0.5'>
              {([['en', l.english], ['ne', l.nepali]] as [string, string][]).map(([val, label]) => (
                <motion.button key={val} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setLanguage(val as Lang)} className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${language === val ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground'}`}>{label}</motion.button>
              ))}
            </div>
          } />
          <SettingRow icon={Palette} label={l.theme} extra={
            <div className='flex bg-secondary rounded-lg p-0.5'>
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setTheme('light')} className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${theme !== 'dark' ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground'}`}><Sun className='w-3 h-3' />{l.light}</motion.button>
              <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => setTheme('dark')} className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${theme === 'dark' ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground'}`}><Moon className='w-3 h-3' />{l.dark}</motion.button>
            </div>
          } />
          <SettingRow icon={Shield} label={l.security} chevron />
          <SettingRow icon={HelpCircle} label={l.helpSupport} chevron />
        </div>
      </CardContent></Card></motion.div>

      <motion.div {...fu(4)} className='pb-2'>
        <motion.button onClick={handleLogout} whileHover={{ scale: 1.02, x: -2 }} whileTap={{ scale: 0.97 }} className='w-full h-10 rounded-xl gap-1.5 text-xs font-medium text-destructive border border-destructive/30 hover:bg-destructive/10 flex items-center justify-center transition-colors'>
          <LogOut className='w-3.5 h-3.5' />{l.logout}
        </motion.button>
      </motion.div>
    </div>
  );
}

function SettingRow({ icon: Icon, label, toggle, toggleOn, onToggle, extra, chevron }: { icon: React.ElementType; label: string; toggle?: boolean; toggleOn?: boolean; onToggle?: () => void; extra?: React.ReactNode; chevron?: boolean }) {
  return (
    <motion.div className='flex items-center justify-between p-2.5 rounded-xl hover:bg-secondary/50 transition-colors cursor-default' whileHover={toggle ? { x: 2 } : { x: 2, scale: 1.005 }} whileTap={toggle ? undefined : { scale: 0.995 }}>
      <div className='flex items-center gap-2.5'>
        <div className='w-7 h-7 rounded-lg bg-secondary flex items-center justify-center'><Icon className='w-3.5 h-3.5' /></div>
        <span className='text-xs font-medium'>{label}</span>
      </div>
      {toggle && <Switch checked={toggleOn} onCheckedChange={onToggle} />}
      {extra}
      {chevron && <ChevronRight className='w-3.5 h-3.5 text-muted-foreground' />}
    </motion.div>
  );
}