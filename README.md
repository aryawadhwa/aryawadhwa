<!-- README.md -->
<div align="center">

<img src="https://raw.githubusercontent.com/aryawadhwa/aryawadhwa/main/assets/banner.svg" alt="Arya Wadhwa" width="100%"/>

<br/>

```css
/* Dark-mode toggle */
<input type="checkbox" id="darkmode" hidden>
<label for="darkmode" id="toggle" title="Toggle dark mode">🌙</label>
<script>
  // Persist dark mode
  const dm = localStorage.getItem('dark');
  if (dm === '1' || (!dm && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
    document.getElementById('darkmode').checked = true;
  }
  document.getElementById('toggle').addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    localStorage.setItem('dark', document.documentElement.classList.contains('dark') ? '1' : '0');
  });
</script>
/* Global styles */
:root { --bg: #fafafa; --text: #1a1a1a; --accent: #ff6b6b; --card: #fff; --shadow: rgba(0,0,0,.08); }
.dark { --bg: #0d1117; --text: #c9d1d9; --accent: #ff6b6b; --card: #161b22; --shadow: rgba(0,0,0,.3); }
html { scroll-behavior: smooth; }
body { background:var(--bg); color:var(--text); font-family:'Inter',system-ui,sans-serif; line-height:1.6; margin:0; padding:0; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.container { max-width:960px; margin:auto; padding:2rem 1rem; }
h1,h2,h3 { margin:1.5rem 0 .5rem; }
<!-- Typed mission -->
<div class="typed">
  <span id="typed"></span>
</div>
<script>
  const txt = "private AI • space • help farmers • see for the blind • dark energy • Stanford";
  let i=0; const speed=80;
  function type(){ if(i<txt.length){ document.getElementById("typed").textContent+=txt.charAt(i); i++; setTimeout(type,speed); } }
  setTimeout(type,500);
</script>
<style>
.typed { font-family:monospace; font-size:1.1rem; color:var(--accent); white-space:nowrap; overflow:hidden; }
</style>
<span title="Python – 95%">🐍 Python</span>
<span title="PyTorch – 85%">🔥 PyTorch</span>
<span title="Transformers – 80%">🤗 Transformers</span>
<span title="Docker – 70%">🐳 Docker</span>
<span title="FastAPI – 75%">⚡ FastAPI</span>
<span title="C – 60%">⚙️ C</span>
<span title="JS/Node – 55%">☕ Node.js</span>
<button onclick="copy('aryawadhwa.ai@rvce.edu.in')">📧 Email</button>
<a href="https://linkedin.com/in/aryawadhwa" target="_blank">💼 LinkedIn</a>
<a href="https://twitter.com/aryawadhwa" target="_blank">🐦 Twitter</a>
<a href="https://github.com/aryawadhwa/aryawadhwa/issues/new">
  <img src="https://img.shields.io/badge/Start%20Conversation-ff6b6b?style=for-the-badge&logo=github" alt="Conversation">
</a>
