# AI-Powered Image Search Application

A sophisticated Streamlit application that enables natural language-based image search and retrieval using Google's Gemini AI. This application provides an intuitive interface for searching through image collections using conversational queries.

## 🌟 Features

- **Natural Language to SQL Conversion**: Advanced NLP capabilities to convert user queries into SQL statements
- **AI-Powered Search**: Leverages Google's Gemini AI for intelligent image matching

- **Advanced Query Processing**:
  - Case-insensitive matching
  - Typo tolerance and fuzzy matching
  - Context-aware query understanding
  - Follow-up question handling

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd nl2sql-image-search-app
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   - Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

4. **Run the Application**
   ```bash
   streamlit run nl2sql_image_search_app.py
   ```

## 📁 Project Structure

```
nl2sql-image-search-app/
├── .streamlit/              # Streamlit configuration
│   └── secrets.toml        # API keys and secrets (gitignored)
├── logs/                   # Application logs
│   └── app.log            # Main log file
├── nl2sql_image_search_app.py     # Main application file
├── data.xlsx              # Image metadata and search data
├── requirements.txt       # Project dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## 🔒 Security

- API keys are stored in `.streamlit/secrets.toml` (gitignored)
- Sensitive data is never committed to version control
- Log files are excluded from git tracking
- Environment variables are properly managed

## 🛠️ Development

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Local Development
1. Fork and clone the repository
2. Create a virtual environment
3. Install dependencies
4. Set up your secrets
5. Run the application locally

## 📦 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect to Streamlit Cloud
3. Configure secrets in Streamlit Cloud dashboard
4. Deploy your application

### Self-Hosting
1. Set up a Python environment on your server
2. Install dependencies
3. Configure secrets
4. Run with Streamlit

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Nachiappan Ravi - nachu.ravi97@gmail.com

Project Link: [https://github.com/NachiappanRavi/nl2sql-image-search-app](https://github.com/NachiappanRavi/nl2sql-image-search-app) "# nl2sql-image-search-app" 
