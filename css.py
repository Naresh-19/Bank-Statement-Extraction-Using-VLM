streamlit_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #1A202C;
        color: white;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .feature-card {
        background: linear-gradient(145deg, #2d3748, #202c3c);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #1A202C;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.15);
    }
    .info-card {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3);
    }
    .success-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.3);
    }
    .error-card {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.3);
    }
    .warning-card {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 14px 0 rgba(245, 158, 11, 0.3);
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .metric-card {
        background: linear-gradient(145deg, #2d3748, #202c3c);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #1A202C;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
        text-align: center;
        min-width: 150px;
        flex: 1;
        max-width: 200px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    .column-tag {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(139, 92, 246, 0.3);
    }
    .main .block-container {
        max-width: 100% !important;
        padding: 1rem 2rem !important;
    }
    .stDataFrame > div {
        width: 100% !important;
    }
    .element-container {
        width: 100% !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px rgba(59, 130, 246, 0.4);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.3);
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px rgba(16, 185, 129, 0.4);
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #1A202C, #1A202C);
    }
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .flex-center {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
"""