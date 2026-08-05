import { create } from 'zustand';
import { mockTransactions, mockContacts, APP_NAME, CURRENCY_CODE, formatCurrency, localName, type Lang, type Bilingual, type Transaction, type Contact } from './wallet-data';

export type WalletTab = 'dashboard' | 'send' | 'receive' | 'transactions' | 'profile';

interface WalletState {
  isLoggedIn: boolean;
  setIsLoggedIn: (v: boolean) => void;
  authMode: 'login' | 'signup';
  setAuthMode: (m: 'login' | 'signup') => void;
  language: Lang;
  setLanguage: (l: Lang) => void;
  activeTab: WalletTab;
  setActiveTab: (tab: WalletTab) => void;
  balance: number;
  transactions: Transaction[];
  contacts: Contact[];
  sendAmount: string;
  setSendAmount: (a: string) => void;
  selectedContact: Contact | null;
  setSelectedContact: (c: Contact | null) => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  txSearchQuery: string;
  setTxSearchQuery: (q: string) => void;
  txFilter: 'all' | 'sent' | 'received' | 'pending';
  setTxFilter: (f: 'all' | 'sent' | 'received' | 'pending') => void;
  isSending: boolean;
  sendMoney: (contact: Contact, amount: number, lang: Lang) => void;
  profileName: Bilingual;
  profileEmail: string;
  profilePhone: string;
  notificationsEnabled: boolean;
  biometricEnabled: boolean;
  toggleNotifications: () => void;
  toggleBiometric: () => void;
}

export const useWalletStore = create<WalletState>((set) => ({
  isLoggedIn: false,
  setIsLoggedIn: (v) => set({ isLoggedIn: v }),
  authMode: 'login',
  setAuthMode: (m) => set({ authMode: m }),
  language: 'ne',
  setLanguage: (l) => set({ language: l }),
  activeTab: 'dashboard',
  setActiveTab: (tab) => set({ activeTab: tab }),
  balance: 345620,
  transactions: mockTransactions,
  contacts: mockContacts,
  sendAmount: '',
  setSendAmount: (a) => set({ sendAmount: a }),
  selectedContact: null,
  setSelectedContact: (c) => set({ selectedContact: c }),
  searchQuery: '',
  setSearchQuery: (q) => set({ searchQuery: q }),
  txSearchQuery: '',
  setTxSearchQuery: (q) => set({ txSearchQuery: q }),
  txFilter: 'all',
  setTxFilter: (f) => set({ txFilter: f }),
  isSending: false,
  sendMoney: (contact, amount, lang) => {
    set({ isSending: true });
    setTimeout(() => {
      const newTx: Transaction = {
        id: String(Date.now()),
        type: 'sent',
        name: contact.name,
        amount,
        currency: CURRENCY_CODE,
        date: new Date().toISOString(),
        description: {
          en: `Payment sent to ${contact.name.en}`,
          ne: `${contact.name.ne} लाई भुक्तानी पठाइयो`,
        },
        status: 'completed',
      };
      set((s) => ({
        transactions: [newTx, ...s.transactions],
        balance: s.balance - amount,
        isSending: false, sendAmount: '', selectedContact: null, activeTab: 'dashboard',
      }));
    }, 1500);
  },
  profileName: { en: 'Ram Bahadur Shrestha', ne: 'राम बहादुर श्रेष्ठ' },
  profileEmail: 'ram.shrestha@gmail.com',
  profilePhone: '+977 9841-000123',
  notificationsEnabled: true,
  biometricEnabled: false,
  toggleNotifications: () => set((s) => ({ notificationsEnabled: !s.notificationsEnabled })),
  toggleBiometric: () => set((s) => ({ biometricEnabled: !s.biometricEnabled })),
}));
