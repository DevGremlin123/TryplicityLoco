import express, { Request, Response } from 'express';
import path from 'path';

const app = express();
const PORT = 5000;

// Model inference server URL (set via env or defaults to localhost)
// Point this to your RunPod pod's inference server
const MODEL_API = process.env.MODEL_API_URL || 'http://localhost:8080';

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

interface ChatResponse {
  id: string;
  text: string;
  sources: string[];
  timestamp: string;
}

const FALLBACK_RESPONSES: { text: string; sources: string[] }[] = [
  {
    text: "That's a fascinating question. Based on current research and available data, there are several key perspectives to consider. The consensus among experts suggests a nuanced approach is most effective, balancing both empirical evidence and practical application.",
    sources: ["arxiv.org/research-insights", "nature.com/perspectives", "scholar.google.com"]
  },
  {
    text: "Great question! Here's what I found: this topic has been extensively studied across multiple disciplines. The short answer is that it depends on context, but the most widely accepted view points toward a combination of factors working together rather than any single cause.",
    sources: ["wikipedia.org/wiki/Overview", "britannica.com/analysis", "sciencedirect.com"]
  },
  {
    text: "I've analyzed several sources on this topic. The key takeaway is that recent developments have significantly shifted our understanding. What was once considered definitive has evolved, and newer frameworks offer more comprehensive explanations.",
    sources: ["reuters.com/technology", "mit.edu/research", "ieee.org/publications"]
  },
  {
    text: "Interesting! Let me break this down. There are essentially three schools of thought on this matter. The first emphasizes structural factors, the second focuses on behavioral aspects, and the third takes a holistic systems-level view. Current evidence supports the integrated approach.",
    sources: ["stanford.edu/papers", "jstor.org/journals", "researchgate.net"]
  },
  {
    text: "Here's a comprehensive overview: this subject sits at the intersection of multiple fields. The most current data indicates a trend toward more sophisticated models that account for previously overlooked variables.",
    sources: ["pubmed.gov/latest", "acm.org/digital-library", "springer.com/articles"]
  },
  {
    text: "After searching through the latest information, here's what stands out: the landscape has changed dramatically in recent years. New methodologies and tools have enabled deeper analysis, revealing patterns that weren't visible before.",
    sources: ["techcrunch.com/analysis", "wired.com/science", "nih.gov/research"]
  },
  {
    text: "The foundational principles are well-established, but application continues to evolve. What makes this particularly interesting is how different contexts lead to different optimal strategies, suggesting there's no one-size-fits-all solution.",
    sources: ["harvard.edu/publications", "oxfordacademic.com", "cambridge.org/core"]
  },
  {
    text: "Excellent question — here's the synthesis of what's currently known: multiple independent studies converge on similar conclusions, lending strong credibility to the prevailing theory. However, there are notable exceptions worth attention.",
    sources: ["pnas.org/research", "frontiersin.org", "biorxiv.org/papers"]
  },
  {
    text: "The short answer is nuanced: think of it as a spectrum rather than a binary. Most real-world scenarios fall somewhere in the middle, and the most successful approaches acknowledge and work with this complexity.",
    sources: ["medium.com/deep-dives", "substack.com/analysis", "arstechnica.com"]
  },
  {
    text: "Here's what the latest research reveals: we're at an inflection point in our understanding of this topic. Traditional models are being augmented with new data-driven insights, creating a richer and more accurate picture.",
    sources: ["nature.com/news", "science.org/advances", "thelancet.com/journals"]
  }
];

async function queryModel(message: string): Promise<{ text: string; sources: string[] } | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);

    const res = await fetch(`${MODEL_API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!res.ok) return null;
    const data = await res.json();
    return { text: data.text, sources: data.sources || [] };
  } catch {
    return null;
  }
}

app.post('/api/chat', async (req: Request, res: Response) => {
  const { message } = req.body;
  if (!message || typeof message !== 'string') {
    res.status(400).json({ error: 'Message is required' });
    return;
  }

  // Try the real model first
  const modelResult = await queryModel(message);

  let text: string;
  let sources: string[];

  if (modelResult) {
    text = modelResult.text;
    sources = modelResult.sources;
  } else {
    // Fallback to canned responses if model is unavailable
    const pick = FALLBACK_RESPONSES[Math.floor(Math.random() * FALLBACK_RESPONSES.length)];
    text = pick.text;
    sources = pick.sources;
  }

  const response: ChatResponse = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2),
    text,
    sources,
    timestamp: new Date().toISOString()
  };
  res.json(response);
});

// Health check that also reports model status
app.get('/api/status', async (_req: Request, res: Response) => {
  let modelOnline = false;
  try {
    const check = await fetch(`${MODEL_API}/health`, { signal: AbortSignal.timeout(3000) });
    modelOnline = check.ok;
  } catch { /* model offline */ }

  res.json({
    status: 'ok',
    model: modelOnline ? 'connected' : 'offline (using fallback)',
    modelApi: MODEL_API,
  });
});

app.listen(PORT, () => {
  console.log(`\n  Tryplicity running → http://localhost:${PORT}`);
  console.log(`  Model API → ${MODEL_API}`);
  console.log(`  Set MODEL_API_URL env var to point to your RunPod inference server\n`);
});
