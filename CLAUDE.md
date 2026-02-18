# Tryplicity — Development Guidelines

## Project
Tryplicity is an AI-powered search/chat interface. Dark theme + purple accent (#8b5cf6). Node.js + Express backend, single-file HTML/CSS/JS frontend. Hosted on port 5000.

## Design Principles
- **Think before acting.** Every UI element should be polished, interactive, and feel like a real product — not a placeholder.
- **No shortcuts.** Buttons should have hover states. Dropdowns should animate. Inputs should have focus rings. Everything should feel intentional.
- **Consistency.** Use the existing CSS variables. Don't introduce new colors. Purple + dark, nothing else.
- **Logo.** Circle-in-circle (hollow outer, solid inner) in #8b5cf6. Used in sidebar, welcome screen, and AI chat avatar.
- **Font.** Inter. Already loaded via Google Fonts.
- **Dark theme.** --bg: #10101f, --white: #1a1a2e (surfaces), --text: #eeeef5, --purple: #7c3aed, --purple-light: #8b5cf6.

## Code Standards
- All frontend is in public/index.html (single file — CSS + HTML + JS).
- Server is server.js (Express, serves static + /api/chat endpoint).
- Local conversation history via localStorage. No database.
- 10 default AI responses with fake sources. No real AI model yet (coming soon).
- Keep code clean and minimal. No libraries, no frameworks. Vanilla only.

## When Making Changes
- Always read the current file state before editing.
- Think about hover states, animations, transitions, accessibility.
- If adding a UI component (dropdown, modal, tooltip), make it fully functional with open/close, click-outside-to-close, keyboard support.
- Match existing spacing, border-radius, and shadow patterns.
