streamlit_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Reset & Base Styles */
    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --dark-gradient: linear-gradient(135deg, #0f1419 0%, #1a202c 25%, #2d3748 100%);
        --glass-bg: rgba(45, 55, 72, 0.85);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        --border-radius-lg: 24px;
        --border-radius-md: 16px;
        --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Main App Container */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--dark-gradient);
        color: var(--text-primary);
        line-height: 1.6;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(245, 158, 11, 0.05) 0%, transparent 50%);
        animation: floatingOrbs 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes floatingOrbs {
        0%, 100% { transform: rotate(0deg) scale(1); }
        33% { transform: rotate(120deg) scale(1.1); }
        66% { transform: rotate(240deg) scale(0.9); }
    }
    
    /* Main Container */
    .main .block-container {
        max-width: 100% !important;
        padding: 2.5rem 3.5rem !important;
        position: relative;
    }
    
    /* Headers - Target Streamlit's actual header elements */
    .stMarkdown h1 {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(2.75rem, 6vw, 4.5rem) !important;
        font-weight: 800 !important;
        background: var(--primary-gradient);
        background-size: 300% 300%;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text;
        text-align: center !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.1 !important;
        letter-spacing: -0.03em !important;
        animation: gradientShift 8s ease-in-out infinite, titleFloat 6s ease-in-out infinite;
        position: relative;
    }
    
    .stMarkdown h1::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: var(--primary-gradient);
        border-radius: 2px;
        animation: underlineExpand 2s ease-out 0.5s both;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 200% 50%; }
        75% { background-position: 100% 50%; }
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes underlineExpand {
        from { width: 0; opacity: 0; }
        to { width: 100px; opacity: 1; }
    }
    
    /* Buttons - Target actual Streamlit button classes */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 1.2rem 3rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        font-family: 'Poppins', sans-serif !important;
        position: relative;
        overflow: hidden;
        transition: var(--transition) !important;
        box-shadow: 
            0 8px 25px rgba(59, 130, 246, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        cursor: pointer !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        min-height: auto !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.6s ease;
        z-index: 1;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 15px 40px rgba(59, 130, 246, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        font-family: 'Poppins', sans-serif !important;
        width: 100% !important;
        transition: var(--transition) !important;
        box-shadow: 
            0 8px 25px rgba(16, 185, 129, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        cursor: pointer !important;
        position: relative;
        overflow: hidden;
        min-height: auto !important;
    }
    
    .stDownloadButton > button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        transform: rotate(45deg) translateX(-100%);
        transition: transform 0.6s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 
            0 12px 35px rgba(16, 185, 129, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        background: linear-gradient(135deg, #059669 0%, #065f46 100%) !important;
    }
    
    .stDownloadButton > button:hover::before {
        transform: rotate(45deg) translateX(100%);
    }
    
    /* Info/Success/Error Messages */
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 50%, #1d4ed8 100%) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 
            0 10px 25px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stInfo::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        animation: shimmerSweep 3s infinite;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #047857 50%, #059669 100%) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 
            0 10px 25px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stSuccess::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        animation: shimmerSweep 3s infinite;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 50%, #dc2626 100%) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 
            0 10px 25px rgba(239, 68, 68, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #b45309 50%, #d97706 100%) !important;
        color: white !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 
            0 10px 25px rgba(245, 158, 11, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    @keyframes shimmerSweep {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    /* Metrics */
    .metric-row {
        display: flex !important;
        gap: 2rem !important;
        margin: 2rem 0 !important;
        flex-wrap: wrap !important;
    }
    
    [data-testid="metric-container"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        border: 1px solid var(--glass-border) !important;
        text-align: center !important;
        position: relative !important;
        transition: var(--transition) !important;
        overflow: hidden !important;
        cursor: pointer !important;
        min-width: 200px !important;
        flex: 1 !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-8px) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        box-shadow: var(--shadow-lg) !important;
        background: rgba(45, 55, 72, 0.95) !important;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.5s ease;
    }
    
    [data-testid="metric-container"]:hover::before {
        transform: scaleX(1);
    }
    
    [data-testid="metric-container"] > div > div {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: var(--primary-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px dashed var(--glass-border) !important;
        border-radius: var(--border-radius-md) !important;
        padding: 3rem 2rem !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stFileUploader > div > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(102, 126, 234, 0.1), 
            transparent);
        transition: left 0.8s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6) !important;
        background: rgba(45, 55, 72, 0.95) !important;
        transform: scale(1.01) !important;
    }
    
    .stFileUploader > div > div:hover::before {
        left: 100%;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: var(--border-radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--glass-border) !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stDataFrame [data-testid="stDataFrame"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-family: inherit !important;
        backdrop-filter: blur(15px) !important;
        transition: var(--transition) !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.7) !important;
        box-shadow: 
            0 0 0 4px rgba(102, 126, 234, 0.1),
            0 4px 20px rgba(102, 126, 234, 0.2) !important;
        background: rgba(45, 55, 72, 1) !important;
        outline: none !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: 10px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stProgress > div > div > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.4), 
            transparent);
        animation: progressShine 2s infinite;
    }
    
    @keyframes progressShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Images */
    .stImage > img {
        border-radius: var(--border-radius-md) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: var(--transition) !important;
    }
    
    .stImage:hover > img {
        transform: scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(15px) !important;
        padding: 1rem 1.5rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(45, 55, 72, 1) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateX(4px) !important;
    }
    
    /* Columns */
    .stColumns {
        gap: 2rem !important;
    }
    
    /* JSON Display */
    .stJson {
        background: var(--glass-bg) !important;
        border-radius: var(--border-radius-md) !important;
        border: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a202c, #2d3748) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem 2rem !important;
        }
        
        .stMarkdown h1 {
            font-size: 2.5rem !important;
        }
        
        [data-testid="metric-container"] {
            min-width: auto !important;
            flex: none !important;
            width: 100% !important;
        }
        
        .metric-row {
            flex-direction: column !important;
        }
    }
    
    /* Enhanced Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(45, 55, 72, 0.6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 10px;
        border: 2px solid rgba(45, 55, 72, 0.6);
        transition: var(--transition);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
        border-color: rgba(45, 55, 72, 0.8);
    }
    
    /* Focus States */
    *:focus {
        outline: 2px solid rgba(102, 126, 234, 0.7) !important;
        outline-offset: 2px !important;
        border-radius: 4px !important;
    }
    
    /* Selection Styles */
    ::selection {
        background: rgba(102, 126, 234, 0.3);
        color: inherit;
    }
    
    /* Page Load Animation */
    .stApp > div {
        animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(40px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
</style>
"""