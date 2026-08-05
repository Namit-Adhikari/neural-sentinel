import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "next-themes";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NeuralSentinel - Fraudulent Transaction Detection System",
  description: "An AI-powered fraudulent transaction detection system for Nepali banking channels.",
  keywords: ["NeuralSentinel", "fraud detection", "Nepal", "banking", "NPR", "transaction security"],
  authors: [{ name: "NeuralSentinel Team" }],
  icons: {},
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "NeuralSentinel",
  },
  manifest: "/manifest.json",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#1a1a1a" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Theme-aware favicon — always starts with light logo, JS switches on class change */}
        <link rel="icon" href="/logo-light.svg" id="dynamic-favicon" />
        <link rel="apple-touch-icon" href="/logo-light.svg" id="dynamic-apple-icon" />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){
              var f=document.getElementById('dynamic-favicon');
              var a=document.getElementById('dynamic-apple-icon');
              function u(){
                var isDark=document.documentElement.classList.contains('dark');
                var href=isDark?'/logo-dark.svg':'/logo-light.svg';
                f.href=href;
                a.href=href;
              }
              u();
              var o=new MutationObserver(u);
              o.observe(document.documentElement,{attributes:true,attributeFilter:['class']});
            })();`,
          }}
        />
        {/* Fullscreen on mobile — hides browser chrome / search bar */}
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        {/* Prevent pull-to-refresh on mobile */}
        <style>{`html { overscroll-behavior: none; }`}</style>
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
