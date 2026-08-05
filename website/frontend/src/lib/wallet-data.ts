export type Bilingual = { en: string; ne: string };
export type Lang = 'en' | 'ne';

export interface Transaction {
  id: string;
  type: 'sent' | 'received' | 'pending';
  name: Bilingual;
  amount: number;
  currency: string;
  date: string;
  description: Bilingual;
  status: 'completed' | 'pending' | 'failed';
}

export interface Contact {
  id: string;
  name: Bilingual;
  phone?: string;
  email?: string;
  isFavorite?: boolean;
}

export const APP_NAME = 'NeuralSentinel';
export const CURRENCY_CODE = 'NPR';

/** Get the localized currency symbol */
export function currencySymbol(lang: Lang): string {
  return lang === 'en' ? 'Rs' : 'रू';
}

/** Format amount with localized currency symbol */
export function formatCurrency(amount: number, lang: Lang = 'ne'): string {
  return `${currencySymbol(lang)} ${amount.toLocaleString('en-US')}`;
}

export function formatDate(dateStr: string, lang: Lang): string {
  const date = new Date(dateStr);
  const now = new Date();
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const n = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const diffDays = Math.floor((n.getTime() - d.getTime()) / 86400000);
  if (diffDays <= 0) return lang === 'ne' ? 'आज' : 'Today';
  if (diffDays === 1) return lang === 'ne' ? 'हिजो' : 'Yesterday';
  if (diffDays < 7) return lang === 'ne' ? `${diffDays} दिन अघि` : `${diffDays} days ago`;
  return date.toLocaleDateString(lang === 'ne' ? 'ne-NP' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function getInitials(name: string): string {
  return name.split(/\s+/).map(w => w[0] || '').join('').toUpperCase().slice(0, 2);
}

export function localName(b: Bilingual, lang: Lang): string {
  return b[lang];
}

export const mockTransactions: Transaction[] = [
  {
    id: '1', type: 'received', name: { en: 'Aarav Shrestha', ne: 'आरव श्रेष्ठ' },
    amount: 25000, currency: CURRENCY_CODE, date: '2026-08-05T10:30:00Z',
    description: { en: 'Monthly rent share for Lalitpur apartment', ne: 'ललितपुरको फ्ल्याटको महिनावारी घर भाडा बाँडणी' },
    status: 'completed',
  },
  {
    id: '2', type: 'sent', name: { en: 'Srijana Gurung', ne: 'श्रिजना गुरुङ' },
    amount: 5500, currency: CURRENCY_CODE, date: '2026-08-04T15:20:00Z',
    description: { en: 'Dinner at Thamel rooftop restaurant', ne: 'ठमेलको रूफटप रेस्टुरेन्टमा साँझको खाजा' },
    status: 'completed',
  },
  {
    id: '3', type: 'sent', name: { en: 'Daraz', ne: 'दाराज' },
    amount: 2100, currency: CURRENCY_CODE, date: '2026-08-04T00:00:00Z',
    description: { en: 'Online medicine order from Daraz', ne: 'दाराजबाट अनलाइन औषधि मंगाइएको' },
    status: 'completed',
  },
  {
    id: '4', type: 'received', name: { en: 'Bikash Thapa', ne: 'विकास थापा' },
    amount: 15000, currency: CURRENCY_CODE, date: '2026-08-03T09:15:00Z',
    description: { en: 'Freelance website design project payment', ne: 'वेबसाइट डिजाइन फ्रिलान्स प्रोजेक्टको भुक्तानी' },
    status: 'completed',
  },
  {
    id: '5', type: 'pending', name: { en: 'Nisha Tamang', ne: 'निशा तामाङ' },
    amount: 8000, currency: CURRENCY_CODE, date: '2026-08-05T12:00:00Z',
    description: { en: 'Kathmandu Jazz Festival ticket booking', ne: 'काठमाडौं ज्याज फेस्टिभलको टिकेट बुकिङ' },
    status: 'pending',
  },
  {
    id: '6', type: 'sent', name: { en: 'Bhatbhateni', ne: 'भटभटेनी' },
    amount: 7350, currency: CURRENCY_CODE, date: '2026-08-02T18:45:00Z',
    description: { en: 'Weekly grocery shopping at Bhatbhateni', ne: 'भटभटेनी सुपरमार्केटमा साप्ताहिक किराना खरिद' },
    status: 'completed',
  },
  {
    id: '7', type: 'received', name: { en: 'Rajesh Poudel', ne: 'राजेश पौडेल' },
    amount: 50000, currency: CURRENCY_CODE, date: '2026-08-01T14:00:00Z',
    description: { en: 'Dashain festival gift from family', ne: 'दशैं पर्वको शुभकामना भेटी परिवारबाट' },
    status: 'completed',
  },
  {
    id: '8', type: 'sent', name: { en: 'Pathao', ne: 'पाथाओ' },
    amount: 350, currency: CURRENCY_CODE, date: '2026-08-01T08:30:00Z',
    description: { en: 'Pathao bike ride to Tribhuvan Airport', ne: 'त्रिभुवन विमानस्थलसम्म पाथाओ बाइक सवारी' },
    status: 'completed',
  },
];

export const mockContacts: Contact[] = [
  { id: '1', name: { en: 'Aarav Shrestha', ne: 'आरव श्रेष्ठ' }, phone: '+977 9841-234567', email: 'aarav.shrestha@gmail.com', isFavorite: true },
  { id: '2', name: { en: 'Srijana Gurung', ne: 'श्रिजना गुरुङ' }, phone: '+977 9851-345678', email: 'srijana.gurung@outlook.com', isFavorite: true },
  { id: '3', name: { en: 'Bikash Thapa', ne: 'विकास थापा' }, phone: '+977 9861-456789', email: 'bikash.thapa@gmail.com', isFavorite: false },
  { id: '4', name: { en: 'Nisha Tamang', ne: 'निशा तामाङ' }, phone: '+977 9842-567890', email: 'nisha.tamang@yahoo.com', isFavorite: true },
  { id: '5', name: { en: 'Rajesh Poudel', ne: 'राजेश पौडेल' }, phone: '+977 9852-678901', email: 'rajesh.poudel@gmail.com', isFavorite: false },
  { id: '6', name: { en: 'Laxmi Maharjan', ne: 'लक्ष्मी महर्जन' }, phone: '+977 9862-789012', email: 'laxmi.maharjan@gmail.com', isFavorite: false },
  { id: '7', name: { en: 'Prabhat Basnet', ne: 'प्रभात बास्नेत' }, phone: '+977 9843-890123', email: 'prabhat.basnet@outlook.com', isFavorite: true },
  { id: '8', name: { en: 'Sunita Rai', ne: 'सुनिता राई' }, phone: '+977 9853-901234', email: 'sunita.rai@gmail.com', isFavorite: false },
];
