'use client';

import { useWalletStore } from '@/lib/wallet-store';
import { LoginScreen } from '@/components/wallet/LoginScreen';
import { WalletContainer } from '@/components/wallet/WalletContainer';
import { PhoneBezel } from '@/components/wallet/PhoneBezel';

export default function Home() {
  const isLoggedIn = useWalletStore((s) => s.isLoggedIn);

  return (
    <div className='w-screen h-screen overflow-hidden bg-white'>
      <PhoneBezel>
        {isLoggedIn ? <WalletContainer /> : <LoginScreen />}
      </PhoneBezel>
    </div>
  );
}
