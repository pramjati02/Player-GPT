const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const coldWarning = document.getElementById('coldWarning');
const suggestionsEl = document.getElementById('suggestions');

let firstMessage = true;

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

// Enter to send, Shift+Enter for newline
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function fillSuggestion(btn) {
  inputEl.value = btn.textContent;
  inputEl.focus();
  suggestionsEl.style.display = 'none';
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="label">${role === 'user' ? 'You' : 'Scout AI'}</div>
    <div class="bubble">${text}</div>
  `;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

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

async function sendMessage() {
  const query = inputEl.value.trim();
  if (!query) return;

  suggestionsEl.style.display = 'none';

  addMessage('user', query);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  if (firstMessage) {
    coldWarning.classList.add('visible');
    firstMessage = false;
  }

  const typing = addTyping();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: query })
    });

    coldWarning.classList.remove('visible');

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    typing.remove();
    addMessage('bot', data.response);
  } catch (err) {
    typing.remove();
    coldWarning.classList.remove('visible');
    addMessage('bot', `Sorry, something went wrong. The server may still be waking up — please try again in a moment.\n\n(${err.message})`);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}