// ── DOM references ────────────────────────────────────────────────────────
const messagesEl = document.getElementById('messages');   // Scrollable message list
const inputEl = document.getElementById('input');         // Textarea
const sendBtn = document.getElementById('send');          // Send button
const coldWarning = document.getElementById('coldWarning'); // Cold start banner
const suggestionsEl = document.getElementById('suggestions'); // Suggestion chips

// Track whether this is the user's first message (to show the cold start warning)
let firstMessage = true;

// ── Textarea auto-resize ──────────────────────────────────────────────────
// Grows the textarea as the user types, up to a max of 120px
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

// ── Keyboard shortcut ─────────────────────────────────────────────────────
// Enter sends the message; Shift+Enter inserts a newline instead
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Suggestion chips ──────────────────────────────────────────────────────
// Clicking a chip fills the textarea with that text and hides the chips
function fillSuggestion(btn) {
  inputEl.value = btn.textContent;
  inputEl.focus();
  suggestionsEl.style.display = 'none';
}

// ── Add a message bubble to the chat ─────────────────────────────────────
// role: 'user' or 'bot' — controls alignment and label text
function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="label">${role === 'user' ? 'You' : 'Scout AI'}</div>
    <div class="bubble">${text}</div>
  `;
  messagesEl.appendChild(div);
  // Auto-scroll to the latest message
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// ── Add the animated typing indicator ────────────────────────────────────
// Shows three bouncing dots while waiting for the API response
function addTyping() {
  const div = document.createElement('div');
  div.className = 'message bot typing';
  div.innerHTML = `
    <div class="label">Scout AI</div>
    <div class="bubble"><span></span><span></span><span></span></div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

// ── Main send handler ─────────────────────────────────────────────────────
async function sendMessage() {
  const query = inputEl.value.trim();
  if (!query) return; // Do nothing if the input is empty

  // Hide suggestion chips after the first message
  suggestionsEl.style.display = 'none';

  // Render the user's message and clear the input
  addMessage('user', query);
  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Disable the send button while waiting for a response
  sendBtn.disabled = true;

  // Show cold start warning only on the very first message
  if (firstMessage) {
    coldWarning.classList.add('visible');
    firstMessage = false;
  }

  // Show typing indicator while the API call is in flight
  const typing = addTyping();

  try {
    // POST the query to FastAPI's /ask endpoint
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: query })
    });

    // Hide the cold start warning once we get any response
    coldWarning.classList.remove('visible');

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    typing.remove();
    addMessage('bot', data.response);

  } catch (err) {
    // On failure, remove typing indicator and show an error message
    typing.remove();
    coldWarning.classList.remove('visible');
    addMessage('bot', `Sorry, something went wrong. The server may still be waking up — please try again in a moment.\n\n(${err.message})`);

  } finally {
    // Always re-enable the send button and return focus to the input
    sendBtn.disabled = false;
    inputEl.focus();
  }
}