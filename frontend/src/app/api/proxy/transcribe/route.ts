import { NextResponse } from 'next/server';

const BACKEND_ENV = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export async function POST(req: Request) {
  const body = await req.arrayBuffer();
  const contentType = req.headers.get('content-type') || '';

  // Try primary backend, then fallback to 127.0.0.1 IPv4 address (helps on some Windows setups)
  const candidates = [BACKEND_ENV, BACKEND_ENV.replace('localhost', '127.0.0.1')].filter(Boolean);

  const errors: any[] = [];
  for (const base of candidates) {
    try {
      const url = `${base.replace(/\/$/, '')}/transcribe`;
      const headers: Record<string, string> = {};
      if (contentType) headers['content-type'] = contentType;

      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: Buffer.from(body),
      });

      const text = await res.text();
      const responseHeaders: Record<string, string> = {};
      res.headers.forEach((v, k) => (responseHeaders[k] = v));
      return new NextResponse(text, { status: res.status, headers: responseHeaders });
    } catch (err: any) {
      errors.push({ backend: base, error: String(err) });
    }
  }

  return new NextResponse(JSON.stringify({ error: 'All backend attempts failed', attempts: errors }), { status: 502 });
}
