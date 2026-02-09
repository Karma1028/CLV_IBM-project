// ==========================================
// MAIN.JS - Enhanced Navigation & Interactions
// ==========================================

let currentChapter = 0;
let visitedChapters = new Set([0]);

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initPageState(); // Ensure landing page shows first
    initParticles();
    initCounters();
    initSliders();
    initPredictionForm();
    initScrollAnimations();
});

// ===== SCROLL ANIMATIONS =====
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target); // Animate only once
            }
        });
    }, observerOptions);

    // Observer will be attached to sections when they are dynamically loaded
    window.observeTraditionalSections = () => {
        const sections = document.querySelectorAll('.trad-section');
        sections.forEach(section => observer.observe(section));
    };
}

// Ensure app always starts on landing page
function initPageState() {
    const landing = document.getElementById('landing');
    const detective = document.getElementById('detective-mode');
    const traditional = document.getElementById('traditional-mode');

    // Force landing page visible
    landing.classList.add('active');
    landing.style.display = 'flex';

    // Force modes hidden
    detective.classList.remove('active');
    detective.style.display = 'none';
    traditional.classList.remove('active');
    traditional.style.display = 'none';
}

// ===== PARTICLE SYSTEM =====
function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    const particleCount = 20; // Reduced for better performance

    for (let i = 0; i < particleCount; i++) {
        createParticle(container, i);
    }
}

function createParticle(container, index) {
    const particle = document.createElement('div');
    particle.className = 'particle';

    const size = Math.random() * 3 + 1;
    const left = Math.random() * 100;
    const delay = (index / 20) * 15;
    const duration = Math.random() * 10 + 10;

    particle.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${left}%;
        animation-delay: ${delay}s;
        animation-duration: ${duration}s;
    `;

    container.appendChild(particle);
}

// ===== ANIMATED COUNTERS =====
function initCounters() {
    const counters = document.querySelectorAll('.counter');

    counters.forEach(counter => {
        const target = parseFloat(counter.dataset.target);
        const duration = 2000;
        const start = performance.now();

        function animate(currentTime) {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = (target * eased).toFixed(1);

            counter.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }

        requestAnimationFrame(animate);
    });
}

// ===== SLIDERS =====
function initSliders() {
    const tenureSlider = document.getElementById('tenure');
    const tenureValue = document.getElementById('tenure-value');

    if (tenureSlider && tenureValue) {
        tenureSlider.addEventListener('input', () => {
            tenureValue.textContent = tenureSlider.value;
        });
    }
}

// ===== MODE SWITCHING =====
function enterMode(mode) {
    console.log('Entering mode:', mode);
    const landing = document.getElementById('landing');
    const detective = document.getElementById('detective-mode');
    const traditional = document.getElementById('traditional-mode');

    // Hide landing
    landing.classList.remove('active');
    landing.style.display = 'none';

    if (mode === 'detective') {
        detective.classList.add('active');
        detective.style.display = 'flex';
        traditional.classList.remove('active');
        traditional.style.display = 'none';
        initDetectiveMode();
    } else {
        traditional.classList.add('active');
        traditional.style.display = 'block';
        detective.classList.remove('active');
        detective.style.display = 'none';
        initTraditionalMode();
    }
}

function goToLanding() {
    console.log('Going to landing');
    const landing = document.getElementById('landing');
    const detective = document.getElementById('detective-mode');
    const traditional = document.getElementById('traditional-mode');

    // Show landing
    landing.classList.add('active');
    landing.style.display = 'flex';

    // Hide both modes
    detective.classList.remove('active');
    detective.style.display = 'none';
    traditional.classList.remove('active');
    traditional.style.display = 'none';
}

// ===== DETECTIVE MODE =====
function initDetectiveMode() {
    currentChapter = 0;
    visitedChapters = new Set([0]);
    loadChapter(0);
    initChapterDots();
    initKeyboardNav();
}

function loadChapter(index) {
    if (!window.chapters || index < 0 || index >= window.chapters.length) return;

    const chapter = window.chapters[index];
    const container = document.getElementById('chapter-container');

    // Fade out
    container.style.opacity = '0';
    container.style.transform = 'translateY(20px)';

    setTimeout(() => {
        container.innerHTML = `
            <div class="chapter-content">
                <span class="chapter-badge">CHAPTER ${chapter.id}</span>
                <h1 class="chapter-title">${chapter.icon} ${chapter.title}</h1>
                <p class="chapter-subtitle">${chapter.subtitle}</p>
                <div class="chapter-body">${chapter.content}</div>
            </div>
        `;

        // Fade in
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';

        updateChapterNav(index);
        initChapterCharts(chapter.id);

        // Scroll to top
        container.scrollTop = 0;
    }, 300);

    currentChapter = index;
    visitedChapters.add(index);
}

function updateChapterNav(index) {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const indicator = document.querySelector('.chapter-current');
    const progressFill = document.getElementById('progress-fill');

    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === window.chapters.length - 1;

    if (indicator) indicator.textContent = `Chapter ${index + 1}`;
    if (progressFill) progressFill.style.width = `${((index + 1) / window.chapters.length) * 100}%`;

    updateChapterDots(index);
}

function initChapterDots() {
    const dotsContainer = document.getElementById('chapter-dots');
    if (!dotsContainer || !window.chapters) return;

    dotsContainer.innerHTML = '';

    window.chapters.forEach((_, i) => {
        const dot = document.createElement('div');
        dot.className = `chapter-dot ${i === 0 ? 'active' : ''}`;
        dot.onclick = () => goToChapter(i);
        dotsContainer.appendChild(dot);
    });
}

function updateChapterDots(index) {
    const dots = document.querySelectorAll('.chapter-dot');
    dots.forEach((dot, i) => {
        dot.className = 'chapter-dot';
        if (i === index) dot.classList.add('active');
        else if (visitedChapters.has(i)) dot.classList.add('completed');
    });
}

function prevChapter() {
    if (currentChapter > 0) loadChapter(currentChapter - 1);
}

function nextChapter() {
    if (currentChapter < window.chapters.length - 1) loadChapter(currentChapter + 1);
}

function goToChapter(index) {
    loadChapter(index);
}

function initKeyboardNav() {
    document.onkeydown = (e) => {
        if (!document.getElementById('detective-mode').classList.contains('active')) return;
        if (document.getElementById('ai-panel').classList.contains('active')) return;
        if (document.getElementById('predictor-modal').classList.contains('active')) return;

        if (e.key === 'ArrowRight' || e.key === ' ') {
            e.preventDefault();
            nextChapter();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            prevChapter();
        } else if (e.key === 'Escape') {
            goToLanding();
        }
    };
}

// ===== TRADITIONAL MODE =====
function initTraditionalMode() {
    const container = document.getElementById('traditional-content');
    const navLinks = document.getElementById('nav-links');

    if (!container || !window.chapters) return;

    // Build content
    let html = '';
    window.chapters.forEach(chapter => {
        html += `
            <section id="trad-${chapter.id}" class="trad-section">
                <span class="chapter-badge">CHAPTER ${chapter.id}</span>
                <h2 class="chapter-title">${chapter.icon} ${chapter.title}</h2>
                <p class="chapter-subtitle">${chapter.subtitle}</p>
                <div class="chapter-body">${chapter.content}</div>
            </section>
        `;
    });
    container.innerHTML = html;

    // Trigger scroll animations
    if (window.observeTraditionalSections) {
        window.observeTraditionalSections();
    }

    // Build nav links
    if (navLinks) {
        navLinks.innerHTML = window.chapters.map(ch =>
            `<a href="#trad-${ch.id}">${ch.title}</a>`
        ).join('');
    }

    // Init charts for all
    window.chapters.forEach(ch => initChapterCharts(ch.id));
}

// ===== CHARTS =====
function initChapterCharts(chapterId) {
    const chartConfig = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: 'rgba(255,255,255,0.7)', family: 'Inter, sans-serif' },
        margin: { t: 40, r: 30, b: 50, l: 60 }
    };

    const config = { responsive: true, displayModeBar: false };

    switch (chapterId) {
        case 3: initClvDistributionChart(chartConfig, config); break;
        case 4: initCorrelationHeatmap(chartConfig, config); break;
        case 6: initModelComparisonChart(chartConfig, config); break;
        case 7: initClusterChart(chartConfig, config); break;
    }
}

function initClvDistributionChart(chartConfig, config) {
    const container = document.getElementById('clv-distribution-chart');
    if (!container) return;

    const trace = {
        x: Array.from({ length: 100 }, () => Math.random() * 15000 + 1000),
        type: 'histogram',
        marker: {
            color: 'rgba(102, 126, 234, 0.7)',
            line: { color: 'rgba(102, 126, 234, 1)', width: 1 }
        },
        nbinsx: 30
    };

    Plotly.newPlot(container, [trace], {
        ...chartConfig,
        title: { text: 'CLV Distribution Across Customers', font: { size: 14 } },
        xaxis: { title: 'Customer Lifetime Value ($)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Frequency', gridcolor: 'rgba(255,255,255,0.05)' }
    }, config);
}

function initDistributionChart(chartConfig, config) {
    // Kept for potential future use or if Ch3 needs multiple charts
    // Currently unused in Ch3 content 
    const container = document.getElementById('premium-distribution-chart');
    if (!container) return;
    // ... code truncated ...
}

function initCorrelationHeatmap(chartConfig, config) {
    // Matches ID in Chapter 4
    const container = document.getElementById('premium-clv-chart');
    if (!container) return;

    const vars = ['CLV', 'Premium', 'Income', 'Tenure', 'Policies'];
    // ...
    const trace = {
        z: [
            [1.0, 0.70, 0.15, 0.35, 0.25],
            [0.70, 1.0, 0.20, 0.30, 0.18],
            [0.15, 0.20, 1.0, 0.10, 0.12],
            [0.35, 0.30, 0.10, 1.0, 0.28],
            [0.25, 0.18, 0.12, 0.28, 1.0]
        ],
        x: vars,
        y: vars,
        type: 'heatmap',
        colorscale: [[0, '#1a1a2e'], [0.5, '#667eea'], [1, '#a855f7']],
        showscale: true
    };

    Plotly.newPlot(container, [trace], {
        ...chartConfig,
        title: { text: 'Feature Correlation Matrix', font: { size: 14 } }
    }, config);
}

function initModelComparisonChart(chartConfig, config) {
    const container = document.getElementById('model-comparison-chart');
    if (!container) return;

    const models = ['Linear', 'Ridge', 'Lasso', 'Elastic', 'RF'];
    const scores = [0.83, 0.84, 0.82, 0.83, 0.89];

    const trace = {
        x: models,
        y: scores,
        type: 'bar',
        marker: {
            color: scores.map(s => s === 0.89 ? 'rgba(52, 211, 153, 0.9)' : 'rgba(102, 126, 234, 0.7)')
        },
        text: scores.map(s => (s * 100).toFixed(1) + '%'),
        textposition: 'auto'
    };

    Plotly.newPlot(container, [trace], {
        ...chartConfig,
        title: { text: 'Model R² Score Comparison', font: { size: 14 } },
        yaxis: { title: 'R² Score', range: [0.7, 1], gridcolor: 'rgba(255,255,255,0.05)' }
    }, config);
}

function initClusterChart(chartConfig, config) {
    const container = document.getElementById('cluster-chart');
    if (!container) return;

    const clusters = [
        { name: 'Steady Eddies', x: Array.from({ length: 50 }, () => Math.random() * 2 + 6), y: Array.from({ length: 50 }, () => Math.random() * 3000 + 7000), color: 'rgba(52, 211, 153, 0.7)' },
        { name: 'High Rollers', x: Array.from({ length: 40 }, () => Math.random() * 2 + 8), y: Array.from({ length: 40 }, () => Math.random() * 4000 + 10000), color: 'rgba(251, 191, 36, 0.7)' },
        { name: 'Riskmakers', x: Array.from({ length: 35 }, () => Math.random() * 3 + 2), y: Array.from({ length: 35 }, () => Math.random() * 2000 + 4000), color: 'rgba(248, 113, 113, 0.7)' },
        { name: 'Fresh Starts', x: Array.from({ length: 45 }, () => Math.random() * 2 + 1), y: Array.from({ length: 45 }, () => Math.random() * 2500 + 3000), color: 'rgba(96, 165, 250, 0.7)' }
    ];

    const traces = clusters.map(c => ({
        x: c.x, y: c.y,
        mode: 'markers',
        type: 'scatter',
        name: c.name,
        marker: { color: c.color, size: 8 }
    }));

    Plotly.newPlot(container, traces, {
        ...chartConfig,
        title: { text: 'Customer Segments by Tenure & CLV', font: { size: 14 } },
        xaxis: { title: 'Tenure (years)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'CLV ($)', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: -0.2 }
    }, config);
}

// ===== AI PANEL =====
function openAIPanel() {
    document.getElementById('ai-panel').classList.add('active');
    document.getElementById('ai-overlay').classList.add('active');
}

function closeAIPanel() {
    document.getElementById('ai-panel').classList.remove('active');
    document.getElementById('ai-overlay').classList.remove('active');
}

function handleAIKeypress(e) {
    if (e.key === 'Enter') sendAIMessage();
}

function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const message = input.value.trim();
    if (!message) return;

    addUserMessage(message);
    input.value = '';

    if (typeof handleUserMessage === 'function') {
        handleUserMessage(message);
    }
}

function addUserMessage(text) {
    const container = document.getElementById('ai-messages');
    const div = document.createElement('div');
    div.className = 'ai-message user';
    div.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content"><p>${text}</p></div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function aiQuickAction(action) {
    if (typeof window[`ai${action.charAt(0).toUpperCase() + action.slice(1)}Action`] === 'function') {
        window[`ai${action.charAt(0).toUpperCase() + action.slice(1)}Action`]();
    }
}

// ===== PREDICTOR MODAL =====
function openPredictorModal() {
    const modal = document.getElementById('predictor-modal');
    modal.classList.add('active');
    modal.style.display = 'flex';
}

function closePredictorModal() {
    const modal = document.getElementById('predictor-modal');
    modal.classList.remove('active');
    modal.style.display = 'none';
}

window.openPredictorModal = openPredictorModal;

function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const income = parseFloat(document.getElementById('income').value);
        const premium = parseFloat(document.getElementById('premium').value);
        const tenure = parseInt(document.getElementById('tenure').value);
        const policies = parseInt(document.getElementById('policies').value);
        const coverage = document.getElementById('coverage').value;
        const vehicle = document.getElementById('vehicle').value;

        // Calculate CLV
        const coverageMult = { basic: 0.8, extended: 1.0, premium: 1.3 }[coverage] || 1;
        const vehicleMult = { sedan: 1.0, suv: 1.1, luxury: 1.4, sports: 1.2 }[vehicle] || 1;

        const clv = Math.round(
            (premium * tenure * 1.2) +
            (income * 0.03) +
            (policies * 500) *
            coverageMult * vehicleMult
        );

        // Determine segment
        const { emoji, name, desc } = getSegment(clv, tenure, premium);

        // Update UI
        document.getElementById('clv-value').textContent = `$${clv.toLocaleString()}`;
        document.getElementById('segment-emoji').textContent = emoji;
        document.getElementById('segment-name').textContent = name;
        document.getElementById('segment-desc').textContent = desc;
        document.getElementById('prediction-result').classList.remove('hidden');

        // Store for AI
        if (typeof storePrediction === 'function') {
            storePrediction({ income, premium, tenure, policies, coverage, vehicle, clv, segment: name });
        }
    });
}

function getSegment(clv, tenure, premium) {
    if (clv > 10000 && premium > 150) {
        return { emoji: '💎', name: 'High Roller', desc: 'Premium customer with exceptional lifetime value. Prioritize retention.' };
    } else if (tenure > 48 && clv > 6000) {
        return { emoji: '🏠', name: 'Steady Eddie', desc: 'Loyal, long-term customer. Perfect for upselling opportunities.' };
    } else if (clv < 4000 && tenure < 12) {
        return { emoji: '🌱', name: 'Fresh Start', desc: 'New customer with growth potential. Focus on engagement.' };
    } else {
        return { emoji: '⚡', name: 'Riskmaker', desc: 'Variable value customer. Monitor for churn signals.' };
    }
}

function generateAIReport() {
    openAIPanel();
    if (typeof generateCLVReport === 'function') {
        generateCLVReport();
    }
}

// Global exports
window.enterMode = enterMode;
window.goToLanding = goToLanding;
window.prevChapter = prevChapter;
window.nextChapter = nextChapter;
window.goToChapter = goToChapter;
window.openAIPanel = openAIPanel;
window.closeAIPanel = closeAIPanel;
window.handleAIKeypress = handleAIKeypress;
window.sendAIMessage = sendAIMessage;
window.aiQuickAction = aiQuickAction;
window.closePredictorModal = closePredictorModal;
window.generateAIReport = generateAIReport;
