'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';

/**
 * Realistic phone bezel that wraps the app content.
 * Desktop: rounded black bezel with equal spacing top/bottom.
 * Mobile: no bezel — the device IS the phone.
 */
export function PhoneBezel({ children }: { children: ReactNode }) {
  return (
    <>
      {/* Desktop: phone with bezel + equal top/bottom padding */}
      <div className='hidden md:flex w-full h-full items-center justify-center bg-white py-6 px-4'>
        <motion.div
          className='relative w-full h-full max-w-[440px]'
          initial={{ opacity: 0, scale: 0.92, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.6, type: 'spring', stiffness: 180 }}
        >
          {/* The phone body: border = bezel */}
          <div className='w-full h-full rounded-[2.5rem] border-[10px] border-neutral-900 bg-neutral-900 overflow-hidden flex flex-col relative shadow-[0_25px_60px_-12px_rgba(0,0,0,0.3)]'>
            {/* Selfie camera notch — small, centered at top */}
            <div className='absolute top-0 left-1/2 -translate-x-1/2 z-50'>
              <div className='w-[72px] h-[22px] bg-neutral-900 rounded-b-2xl flex items-center justify-center gap-2'>
                <div className='w-[8px] h-[8px] rounded-full bg-neutral-800 ring-1 ring-neutral-700' />
              </div>
            </div>

            {/* Status bar */}
            <StatusBar />

            {/* Screen content */}
            <div className='flex-1 overflow-hidden bg-background relative'>
              {children}
            </div>

            {/* Home indicator */}
            <div className='shrink-0 h-[14px] bg-background flex items-center justify-center'>
              <div className='w-[100px] h-[3.5px] bg-foreground/20 rounded-full' />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Mobile: no bezel, full bleed */}
      <div className='md:hidden w-full h-full overflow-hidden bg-background'>
        {children}
      </div>
    </>
  );
}

/** Phone status bar with time, wifi, battery */
function StatusBar() {
  return (
    <div className='shrink-0 h-[28px] bg-background flex items-center justify-between px-8 text-[10px] font-semibold text-foreground relative z-40'>
      <span className='tabular-nums'>9:41</span>
      <div className='flex items-center gap-1.5'>
        {/* Signal bars */}
        <svg className='w-3.5 h-3' viewBox='0 0 16 10' fill='currentColor'>
          <rect x='0' y='3' width='2.5' height='7' rx='0.5' opacity='0.4' />
          <rect x='4' y='1.5' width='2.5' height='8.5' rx='0.5' opacity='0.6' />
          <rect x='8' y='0.5' width='2.5' height='9.5' rx='0.5' opacity='0.8' />
          <rect x='12' y='0' width='2.5' height='10' rx='0.5' />
        </svg>
        {/* Wifi */}
        <svg className='w-3.5 h-3' viewBox='0 0 14 10' fill='currentColor'>
          <path d='M7 1.5C5 1.5 3.2 2.2 1.8 3.4L1 2.4C2.7 0.9 4.7 0 7 0s4.3 0.9 6 2.4l-0.8 1C10.8 2.2 9 1.5 7 1.5z' opacity='0.4' />
          <path d='M7 4C5.8 4 4.7 4.4 3.8 5.1L3 4.2C4.1 3.3 5.5 2.8 7 2.8s2.9 0.5 4 1.4l-0.8 0.9C9.3 4.4 8.2 4 7 4z' opacity='0.7' />
          <circle cx='7' cy='8' r='1.5' />
        </svg>
        {/* Battery */}
        <div className='w-[20px] h-[10px] rounded-[2.5px] border border-foreground/60 p-[1.5px] relative'>
          <div className='h-full w-3/4 rounded-[1px] bg-emerald-500' />
          <div className='absolute -right-[2px] top-1/2 -translate-y-1/2 w-[1.5px] h-[4px] rounded-r-sm bg-foreground/60' />
        </div>
      </div>
    </div>
  );
}
