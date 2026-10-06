import type { AppProps } from 'next/app';
import Head from 'next/head';
import '../styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>NEXUS-SLM — Neural Support Intelligence Dashboard</title>
        <meta
          name="description"
          content="NEXUS-SLM: Neural EXecution for Unified Support Intelligence — Multilingual Customer Support Ticket Triage"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
