'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useTheme } from 'next-themes';
import Image from 'next/image';

interface LogoProps {
  size?: number;
  className?: string;
  animate?: boolean;
}

/**
 * NeuralSentinel Logo — uses theme-aware light/dark SVG files.
 * Uses `mounted` guard to prevent hydration mismatch from unresolved theme.
 */
export function Logo({ size = 56, className = '', animate = true }: LogoProps) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const src = mounted && resolvedTheme === 'dark' ? '/logo-dark.svg' : '/logo-light.svg';

  const wrapper = animate
    ? { whileHover: { scale: 1.08, rotate: [0, -3, 3, 0] }, transition: { duration: 0.5 } }
    : {};

  return (
    <motion.div className={`flex items-center justify-center ${className}`} {...wrapper}>
      <Image
        src={src}
        alt='NeuralSentinel'
        width={size}
        height={size}
        draggable={false}
        priority={false}
      />
    </motion.div>
  );
}
