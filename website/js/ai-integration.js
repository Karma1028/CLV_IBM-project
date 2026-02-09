// ==========================================
// AI INTEGRATION - OpenRouter API
// ==========================================

const AI_CONFIG = {
    apiKey: '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',
    baseUrl: 'https://openrouter.ai/api/v1/chat/completions',
    models: [
        'arcee-ai/trinity-large-preview:free',
        'tngtech/tng-r1t-chimera:free',
        'tngtech/deepseek-r1t2-chimera:free'
    ],
    currentModelIndex: 0
};

// Store the last prediction for AI context
let lastPrediction = null;

// Set prediction context
function setPredictionContext(prediction) {
    lastPrediction = prediction;
}

// Get the next model (rotate through models for reliability)
function getNextModel() {
    const model = AI_CONFIG.models[AI_CONFIG.currentModelIndex];
    AI_CONFIG.currentModelIndex = (AI_CONFIG.currentModelIndex + 1) % AI_CONFIG.models.length;
    return model;
}

// Make AI request
async function makeAIRequest(messages, retries = 3) {
    for (let attempt = 0; attempt < retries; attempt++) {
        const model = getNextModel();

        try {
            const response = await fetch(AI_CONFIG.baseUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${AI_CONFIG.apiKey}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': window.location.href,
                    'X-Title': 'CLV Case Files'
                },
                body: JSON.stringify({
                    model: model,
                    messages: messages,
                    temperature: 0.7,
                    max_tokens: 1500
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            return data.choices[0].message.content;
        } catch (error) {
            console.error(`Attempt ${attempt + 1} failed with model ${model}:`, error);
            if (attempt === retries - 1) {
                throw error;
            }
        }
    }
}

// Generate CLV Report
async function generateCLVReport() {
    if (!lastPrediction) {
        showAIMessage('assistant', 'Please make a CLV prediction first using the calculator, then I can generate a personalized report!');
        return;
    }

    showAIMessage('assistant', '🔄 Generating your personalized CLV report...');

    const systemPrompt = `You are an expert insurance analytics consultant analyzing Customer Lifetime Value (CLV) predictions. 
You provide actionable, strategic insights based on customer data. Be professional but engaging, using clear headings and bullet points.
Format your response in markdown for readability.`;

    const userPrompt = `Generate a comprehensive CLV analysis report for this customer:

**Prediction Results:**
- Predicted CLV: $${lastPrediction.clv.toLocaleString()}
- Customer Segment: ${lastPrediction.segment}
- Income: $${lastPrediction.inputs.income.toLocaleString()}
- Monthly Premium: $${lastPrediction.inputs.premium}
- Tenure: ${lastPrediction.inputs.tenure} months
- Number of Policies: ${lastPrediction.inputs.policies}
- Coverage Type: ${lastPrediction.inputs.coverage}
- Vehicle Class: ${lastPrediction.inputs.vehicle}

Please provide:
1. **Customer Profile Analysis** - Key insights about this customer type
2. **CLV Interpretation** - What this value means for the business
3. **Risk Assessment** - Potential risks and mitigation strategies
4. **Strategic Recommendations** - 3-4 actionable strategies specific to this segment
5. **Cross-sell/Upsell Opportunities** - Revenue growth potential
6. **Retention Priority** - How to prioritize this customer

Keep the analysis practical and actionable for a business user.`;

    try {
        const response = await makeAIRequest([
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
        ]);

        showAIMessage('assistant', response, true);
    } catch (error) {
        showAIMessage('assistant', '❌ I encountered an error generating the report. Please try again in a moment.');
        console.error('AI Report Error:', error);
    }
}

// Explain the prediction
async function explainPrediction() {
    if (!lastPrediction) {
        showAIMessage('assistant', 'Please make a CLV prediction first, then I can explain how we arrived at that value!');
        return;
    }

    showAIMessage('assistant', '🔄 Analyzing the prediction factors...');

    const systemPrompt = `You are a data scientist explaining a machine learning prediction in simple terms.
Use analogies and clear language. Format in markdown with headers and bullet points.`;

    const userPrompt = `Explain how we predicted a CLV of $${lastPrediction.clv.toLocaleString()} for this customer:

- Income: $${lastPrediction.inputs.income.toLocaleString()}
- Monthly Premium: $${lastPrediction.inputs.premium}
- Tenure: ${lastPrediction.inputs.tenure} months
- Number of Policies: ${lastPrediction.inputs.policies}
- Coverage Type: ${lastPrediction.inputs.coverage}
- Vehicle Class: ${lastPrediction.inputs.vehicle}
- Assigned Segment: ${lastPrediction.segment}

Our model uses Gradient Boosting with these top feature importances:
1. Monthly Premium (38.2%)
2. Number of Policies (18.7%)
3. Total Claim Amount (14.3%)
4. Policy Tenure (11.8%)
5. Vehicle Class (8.4%)

Explain:
1. Why this CLV value makes sense given the inputs
2. Which factors most influenced this prediction
3. How changing key inputs would affect the CLV
4. Why this customer was classified in the "${lastPrediction.segment}" segment`;

    try {
        const response = await makeAIRequest([
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
        ]);

        showAIMessage('assistant', response, true);
    } catch (error) {
        showAIMessage('assistant', '❌ Error explaining prediction. Please try again.');
    }
}

// Strategic advice
async function getStrategicAdvice() {
    if (!lastPrediction) {
        showAIMessage('assistant', 'Please make a CLV prediction first to get personalized strategic advice!');
        return;
    }

    showAIMessage('assistant', '🔄 Formulating strategic recommendations...');

    const systemPrompt = `You are a senior insurance strategy consultant. 
Provide bold, actionable recommendations. Be specific with numbers and timelines where possible.
Format in markdown.`;

    const userPrompt = `Provide strategic recommendations for this customer in the "${lastPrediction.segment}" segment:

**Customer Profile:**
- CLV: $${lastPrediction.clv.toLocaleString()}
- Income: $${lastPrediction.inputs.income.toLocaleString()}
- Premium: $${lastPrediction.inputs.premium}/month
- Tenure: ${lastPrediction.inputs.tenure} months
- Policies: ${lastPrediction.inputs.policies}
- Coverage: ${lastPrediction.inputs.coverage}
- Vehicle: ${lastPrediction.inputs.vehicle}

**Segment Characteristics (${lastPrediction.segment}):**
${getSegmentContext(lastPrediction.segment)}

Please provide:
1. **Immediate Actions** (Next 30 days)
2. **Short-term Strategy** (3-6 months)
3. **Long-term Value Maximization** (1+ year)
4. **Risk Mitigation Steps**
5. **Key Performance Indicators to Track**

Be specific and actionable - what exactly should the account manager do?`;

    try {
        const response = await makeAIRequest([
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
        ]);

        showAIMessage('assistant', response, true);
    } catch (error) {
        showAIMessage('assistant', '❌ Error getting strategic advice. Please try again.');
    }
}

// Get segment context
function getSegmentContext(segment) {
    const contexts = {
        'Steady Eddie': 'Middle-income families, basic coverage, low claims, 31% of portfolio, avg CLV $7,234',
        'High Roller': 'High-income, premium coverage, luxury vehicles, 18% of portfolio, avg CLV $14,892',
        'Riskmaker': 'Higher claim frequency, sports cars, younger demographic, 29% of portfolio, avg CLV $5,621',
        'Fresh Start': 'New customers, first-time buyers, limited history, 22% of portfolio, avg CLV $6,487'
    };
    return contexts[segment] || 'Unknown segment profile';
}

// Handle custom AI chat
async function sendCustomMessage(userMessage) {
    showAIMessage('user', userMessage);
    showAIMessage('assistant', '🔄 Thinking...');

    const systemPrompt = `You are the CLV AI Assistant for an insurance analytics platform. 
You have expertise in Customer Lifetime Value analysis, insurance industry, and data science.
${lastPrediction ? `The user's last CLV prediction was $${lastPrediction.clv.toLocaleString()} in the ${lastPrediction.segment} segment.` : ''}
Be helpful, concise, and professional. Format responses in markdown when appropriate.`;

    try {
        const response = await makeAIRequest([
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userMessage }
        ]);

        // Remove the "Thinking..." message and show response
        removeLastAIMessage();
        showAIMessage('assistant', response, true);
    } catch (error) {
        removeLastAIMessage();
        showAIMessage('assistant', '❌ Sorry, I encountered an error. Please try again.');
    }
}

// UI Helper Functions
function showAIMessage(role, content, isMarkdown = false) {
    const messagesContainer = document.getElementById('ai-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `ai-message ${role}`;

    if (isMarkdown && role === 'assistant') {
        messageDiv.innerHTML = marked.parse(content);
    } else {
        messageDiv.innerHTML = `<p>${content}</p>`;
    }

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLastAIMessage() {
    const messagesContainer = document.getElementById('ai-messages');
    const messages = messagesContainer.querySelectorAll('.ai-message');
    if (messages.length > 0) {
        messages[messages.length - 1].remove();
    }
}

// Quick action handlers
function aiQuickAction(action) {
    switch (action) {
        case 'report':
            generateCLVReport();
            break;
        case 'explain':
            explainPrediction();
            break;
        case 'strategy':
            getStrategicAdvice();
            break;
    }
}

// Handle Enter key in chat input
function handleAIKeypress(event) {
    if (event.key === 'Enter') {
        sendAIMessage();
    }
}

// Send message function
function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const message = input.value.trim();
    if (message) {
        input.value = '';
        sendCustomMessage(message);
    }
}

// Export functions
// Export functions globally for HTML handlers and main.js
window.setPredictionContext = setPredictionContext;
window.generateCLVReport = generateCLVReport;
window.explainPrediction = explainPrediction;
window.getStrategicAdvice = getStrategicAdvice;
window.sendCustomMessage = sendCustomMessage;
window.aiQuickAction = aiQuickAction;
window.handleAIKeypress = handleAIKeypress;
window.sendAIMessage = sendAIMessage;

// Keep namespace for cleaner internal usage if needed
window.AI = {
    setPredictionContext,
    generateCLVReport,
    explainPrediction,
    getStrategicAdvice,
    sendCustomMessage,
    aiQuickAction,
    handleAIKeypress,
    sendAIMessage
};
