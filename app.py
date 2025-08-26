import streamlit as st
from PIL import Image

# Load assets
profile_image = Image.open("assets/profile.jpg")

# Page config
st.set_page_config(page_title="AI Engineer Portfolio", layout="wide")

# Sidebar
with st.sidebar:
    st.image(profile_image, width=150)
    st.title("SADIQ ALI H")
    st.markdown("**AI Engineer | Data Scientist**")
    st.markdown("**Coimbatore**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/sadiq786/)")
    st.markdown("[GitHub](https://github.com/Sadiqali78)")
   #st.markdown("[Instagram](https://your-research-link.com)")
    st.download_button("📄 Download Resume", open("data/SADIQ ALI.pdf", "rb").read(), file_name="SADIQ ALI.pdf", mime="application/pdf")

# About Me
st.header("🧑‍🎓 About Me")
st.write("""
I am an MCA graduate in Artificial Intelligence and Data Science, with hands-on experience in Agentic AI, Generative AI, and Data Science. I specialize in building AI-driven solutions that enhance decision-making and address real-world challenges.""")

# Projects
st.header("💼 Projects")
projects = [
    {
    "title": "AI-Powered Data Analyzer Dashboard",
    "description": """AI Data Analyzer: Intelligent Dashboard with Gemini Agents  
Overview:  
Developed an AI-driven interactive dashboard that transforms raw datasets into insightful visualizations and AI-generated summaries, making data exploration seamless for both technical and non-technical users.

Technologies Used:  
Python, Streamlit, Plotly, Seaborn, Pandas, Scikit-learn, Google Gemini LLM

Approach:  
- Implemented automated data cleaning and label encoding using agent-based pipelines  
- Integrated Google Gemini LLM to generate dataset summaries, KPI interpretations, and chart insights  
- Designed customizable editors for dynamic chart generation (bar, donut, boxplot, scatter matrix, heatmap)  
- Built a Power BI–style dashboard within Streamlit for real-time interactive exploration  

Key Contributions:  
- Engineered end-to-end data analysis pipeline from CSV upload to AI-driven insights  
- Developed reusable agents for data preprocessing, visualization, and interpretation  
- Enabled flexible dashboard customization through interactive column selection  
- Combined traditional statistical methods with generative AI for explainable insights  

Outcome:  
Delivered a real-time AI-powered dashboard that automatically cleans data, generates KPIs, and provides AI-explained visual insights—enhancing decision-making and accessibility in data analytics.
""",

    "image": "assets/DataAnalyser.png",
    "github": "https://github.com/Sadiqali78/AI-Powered-Dashboard-Creation"
},
    {
    "title": "AI-Powered Virtual Courtroom Simulation",
    "description": """AI-Virtual Courtroom Simulation  
Overview:  
Developed an advanced multi-agent AI system that simulates courtroom proceedings by integrating client inputs, legal research, logical analysis, professional legal advice, opposing arguments, and final judicial verdicts. The project demonstrates the application of AI in legal reasoning and decision-making.

Technologies Used:  
Python, Streamlit, Large Language Models (LLM), Generative AI, Multi-Agent Systems, NLP

Approach:  
- Designed specialized AI agents to represent different legal roles (Client, Law Researcher, Logic Analyzer, Legal Advisor, Opponent Lawyer, and Judge)  
- Integrated LLM-powered reasoning for law retrieval, case analysis, and advice generation  
- Enabled simulation of real courtroom dynamics with counter-arguments and verdict generation  
- Developed an interactive Streamlit interface for document uploads, case extraction, and step-by-step courtroom flow

Key Contributions:  
- Implemented the **Client Agent** to process and present case inputs in structured form  
- Built the **Law Research Agent** to fetch and simplify relevant laws, sections, and precedents  
- Developed the **Logic Analyzer Agent** to validate arguments and highlight strengths/weaknesses  
- Engineered the **Legal Advisor Agent** to provide actionable, client-friendly advice  
- Designed the **Opponent Lawyer Agent** to generate realistic counter-arguments  
- Created the **AI Judge Agent** to synthesize all perspectives and deliver a balanced verdict  
- Orchestrated the entire multi-agent pipeline in Streamlit for smooth end-to-end user experience

Outcome:  
Delivered an **AI-powered legal simulation platform** that mirrors courtroom reasoning by combining law research, logical evaluation, and judicial decision-making. This project demonstrates the potential of AI multi-agent systems in the legal domain, making complex legal analysis accessible and interactive.
""",

    "image": "assets/VirtualCourtroom.png",
    "github": "https://github.com/Sadiqali78/AI-Powered-Courtroom-Simulation"
},


    {
        "title": "AI-Powered Fitness Chatbot",
        "description": """FitBot: AI-Powered Fitness Chatbot  
Overview:  
Developed an intelligent, voice-enabled chatbot designed to deliver personalized fitness and nutrition guidance using cutting-edge AI technologies.

Technologies Used:  
Python, Flask, Replit, Voiceflow, Large Language Models (LLM), Generative AI, Retrieval-Augmented Generation (RAG)

Approach:  
- Integrated LLMs with RAG pipelines to generate context-aware responses  
- Implemented advanced NLP techniques for conversational understanding  
- Enabled dynamic and natural interactions using generative AI models  
- Developed a speech interface for real-time voice-based user engagement

Key Contributions:  
- Engineered end-to-end chatbot pipeline from user input to AI-driven response  
- Customized recommendations based on user fitness goals and dietary needs  
- Designed and deployed the system on Replit with a smooth voice interface using Voiceflow  
- Optimized performance to ensure fast, accurate, and interactive feedback

Outcome:  
Delivered a real-time, AI-powered fitness assistant that allows users to engage via voice and receive tailored workout and meal plans—enhancing accessibility and user experience in the health and wellness domain.
""",

        "image": "assets/FitBot.jpg",
        "github": "https://github.com/Sadiqali78/FitBot"
    },
    {
        "title": "Chronic Kidney Disease Prediction",
        "description": """
**Overview:**  
Developed a machine learning model to predict Chronic Kidney Disease based on clinical data, aiming to support early diagnosis and timely medical treatment.

**Technologies Used:**  
Python, NumPy, Pandas, Scikit-learn

**Approach:**  
- Used supervised learning techniques with Gradient Boosting Classifier  
- Applied StandardScaler for feature scaling  
- Tuned model using GridSearchCV  
- Evaluated with accuracy, precision, recall, and F1-score

**Key Contributions:**  
- Cleaned and preprocessed real-world data  
- Performed feature selection and model tuning  
- Achieved 100% accuracy on test data

**Outcome:**  
Delivered an accurate prediction model to assist healthcare professionals in early CKD detection.
""",
        "image": "assets/kidney.jpg",
        "github": "https://github.com/Sadiqali78/HopeAI-/tree/main/MACHINE%20LEARNING/chronic%20kidney%20prediction%20ml%20clsf"
    },
    {
        "title": "Heart Attack Disease Prediction",
        "description": """Heart Attack Disease Prediction  
Overview:  
Developed a machine learning model to predict the risk of heart attacks based on patient clinical data, enabling early diagnosis and proactive treatment planning.

Technologies Used:  
Python, NumPy, Pandas, Scikit-learn

Approach:  
- Used Random Forest Classifier for prediction  
- Applied SelectKBest for feature selection  
- Conducted univariate and bivariate analysis for feature insights  
- Evaluated model performance using a confusion matrix and classification metrics

Key Contributions:  
- Cleaned and preprocessed real-world health data  
- Selected top features based on relevance and performance  
- Trained and validated multiple models to ensure accuracy  
- Achieved reliable classification results with balanced performance across metrics

Outcome:  
Delivered a robust prediction system that supports healthcare professionals in identifying high-risk patients and initiating timely medical interventions.
""",

        "image": "assets/heart.png",
        "github": "https://github.com/Sadiqali78/Heart-attack-prediction/tree/main/heart%20attack%20prediction"
    },
]

for project in projects:
    with st.expander(project["title"]):
        st.image(project["image"], width=400)
        st.write(project["description"])
        st.markdown(f"[GitHub Repository]({project['github']})")

# Certifications
st.header("🎓 Certifications & Achievements")
certificates = [
    {
        "title": "Deloitte Australia - Data Analytics Job Simulation",
        "issuer": "Forage",
        "Certificate ID": "JqRm5kYFufyRWxF4Z",
        "link": "https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/9PBTqmSxAf6zZTseP/io9DzWKe3PTsiS6GG_9PBTqmSxAf6zZTseP_Qvq8G5kdaW3mjY9Jr_1750337008048_completion_certificate.pdf"
    }
]

for cert in certificates:
    with st.expander(cert["title"]):
        st.write(f"**Issuer:** {cert['issuer']}")
        st.write(f"**Certificate ID:** {cert['Certificate ID']}")
        st.markdown(f"[View Certificate]({cert['link']})")

# Professional Skill Set
st.header("🛠️ Professional Skill Set")

# Programming Languages
with st.expander("🧑‍💻 Programming Languages"):
    st.markdown("""
- Python  
- SQL  
""")

# Libraries & Tools
with st.expander("🧰 Libraries & Tools"):
    st.markdown("""
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- SciPy  
- Tableau  
- Power BI 
- Looker Studio 
- Streamlit
""")

# Machine Learning
with st.expander("🤖 Machine Learning"):
    st.markdown("""
- Supervised Learning (Regression, Classification)  
- Unsupervised Learning (Clustering)  
- Model development & evaluation  
- Cross-validation  
- Feature selection & dimensionality reduction  
- Hyperparameter optimization  
""")

# Data Science & Analytics
with st.expander("📊 Data Science & Analytics"):
    st.markdown("""
- Data collection, integration, and management  
- Data cleaning, preprocessing, and transformation  
- Exploratory Data Analysis (EDA) to uncover insights  
- Feature engineering to enhance model performance  
- Statistical modeling and hypothesis testing  
- Data visualization and dashboarding (Tableau, Power BI)  
- Business Intelligence and predictive analytics for decision-making  
""")

# Deep Learning
with st.expander("🧠 Deep Learning"):
    st.markdown("""
- Frameworks & Libraries: TensorFlow, Keras, PyTorch, OpenCV  
- Artificial Neural Networks (ANN)  
- Convolutional Neural Networks (CNN)  
- Recurrent Neural Networks (RNN)  
- Long Short-Term Memory (LSTM)  
- Transformers & attention-based models  
""")


# NLP & Generative AI
with st.expander("📚 Natural Language Processing & Generative AI"):
    st.markdown("""
- Text preprocessing & representation (tokenization, embeddings)  
- Prompt engineering for LLM optimization  
- Large Language Models (LLMs) & Generative AI  
- Retrieval-Augmented Generation (RAG)  
- Agentic AI (AI Agents, AutoGen)  
- Frameworks & Tools: LangChain, Hugging Face Transformers  
- Cloud-based Generative AI: Amazon Bedrock  
""")

# Cloud Platforms
with st.expander("☁️ Cloud Platforms"):
    st.markdown("""
- Google Cloud Platform (GCP)  
- Amazon Web Services (AWS):  
  - EC2 (Compute)  
  - S3 (Storage)  
  - Lambda (Serverless Computing)  
  - SageMaker (Machine Learning)  
  - Bedrock (Generative AI Services)  
""")


# Contact
st.header("✉️ Contact")
st.write("📧 Email: sadiqalisadiq786h@gmail.com")
st.write("📱 Phone: +91-9360329836")