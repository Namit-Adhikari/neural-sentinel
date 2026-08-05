'use client';

import { useState } from 'react';
import { useWalletStore } from '@/lib/wallet-store';
import { t, type Lang } from '@/lib/translations';
import { Logo } from './Logo';
import { Eye, EyeOff, ArrowRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export function LoginScreen() {
  const { authMode, setAuthMode, setIsLoggedIn, language, setLanguage } = useWalletStore();
  const isSignup = authMode === 'signup';
  const [showPw, setShowPw] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const l = t[language];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    setTimeout(() => setIsLoggedIn(true), 1200);
  };

  const inputCls =
    'w-full h-11 px-4 rounded-xl bg-secondary/60 border border-border text-sm outline-none focus:ring-2 focus:ring-primary/30 transition-all placeholder:text-muted-foreground/50';

  return (
    <div className='flex flex-col items-center justify-center h-full px-6 relative'>
      {/* Language toggle - top right */}
      <motion.button
        onClick={() => setLanguage(language === 'en' ? 'ne' : 'en')}
        className='absolute top-2 right-3 px-2.5 py-1 rounded-lg bg-secondary/60 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors'
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92 }}
      >
        {language === 'en' ? 'नेपाली' : 'English'}
      </motion.button>

      <motion.div className='w-full max-w-[280px]' initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5, type: 'spring', stiffness: 200 }}>
        {/* Logo & Brand */}
        <div className='flex flex-col items-center mb-7'>
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, type: 'spring' }}>
            <Logo size={60} />
          </motion.div>
          <motion.h1 className='text-2xl font-bold tracking-tight mt-3' initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            {l.appName}
          </motion.h1>
          <motion.p className='text-[11px] text-muted-foreground mt-0.5' initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            {l.appTagline}
          </motion.p>
        </div>

        <form onSubmit={handleSubmit} className='space-y-2.5'>
          <AnimatePresence mode='wait'>
            {isSignup && (
              <motion.div key='name' initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.25 }}>
                <input type='text' placeholder={l.fullName} value={name} onChange={(e) => setName(e.target.value)} className={inputCls} autoComplete='name' />
              </motion.div>
            )}
          </AnimatePresence>

          <motion.input type='email' placeholder={l.email} value={email} onChange={(e) => setEmail(e.target.value)} className={inputCls} initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} autoComplete='email' />

          <div className='relative'>
            <motion.input type={showPw ? 'text' : 'password'} placeholder={l.password} value={password} onChange={(e) => setPassword(e.target.value)} className={`${inputCls} pr-11`} initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.35 }} autoComplete={isSignup ? 'new-password' : 'current-password'} />
            <motion.button type='button' onClick={() => setShowPw(!showPw)} className='absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors' tabIndex={-1} whileHover={{ scale: 1.15 }} whileTap={{ scale: 0.9 }}>
              {showPw ? <EyeOff className='w-4 h-4' /> : <Eye className='w-4 h-4' />}
            </motion.button>
          </div>

          <motion.button type='submit' disabled={loading || !email || !password} className='w-full h-11 bg-primary text-primary-foreground rounded-xl font-semibold text-sm flex items-center justify-center gap-2 hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50' initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} whileHover={{ scale: 1.02, y: -1 }} whileTap={{ scale: 0.97 }}>
            {loading ? (
              <motion.div className='w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full' animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 0.8, ease: 'linear' }} />
            ) : (
              <>{isSignup ? l.signup : l.login}<ArrowRight className='w-4 h-4' /></>
            )}
          </motion.button>
        </form>

        <motion.div className='mt-4 text-center' initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
          <p className='text-xs text-muted-foreground'>
            {isSignup ? l.alreadyHaveAccount : l.dontHaveAccount}{' '}
            <motion.button onClick={() => setAuthMode(isSignup ? 'login' : 'signup')} className='text-primary font-semibold hover:underline' whileHover={{ scale: 1.05 }}>
              {isSignup ? l.loginLink : l.signupLink}
            </motion.button>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}