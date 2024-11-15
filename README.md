# 🎭 TweetFeel: Sentiment Analysis with Style

## 🤔 Overview
TweetFeel is a sophisticated sentiment analysis tool that decodes Twitter's emotional landscape. Using machine learning and natural language processing, it analyzes tweets to determine sentiment patterns - because sometimes you need more than just a like button to understand what people really think.

## 🌟 Key Features
- 📱 Real-time tweet collection and analysis
- 🧼 Advanced text preprocessing and cleaning
- 🎯 Multi-class sentiment classification
- 🤖 Machine Learning pipeline with Logistic Regression
- 📈 Data visualization and analytics
- ⌨️ CLI interface for seamless interaction
- 🐳 Containerized with Docker for consistent deployment

## 🚀 Getting Started

### Prerequisites
- Docker (for containerized deployment)
- Twitter Developer Account credentials
- Git

### Installation & Setup

1. Clone the repository

bash
git clone https://github.com/McMarian/TweetFeel.git
cd TweetFeel

2. Create a `.env` file with your Twitter API credentials:

bash
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret


3. Launch with Docker

bash
docker-compose up --build

## 🗂️ Project Structure
TweetFeel/
├── main.py # Application entry point
├── data_collection.py # Twitter API integration
├── data_preprocessing.py# Text preprocessing pipeline
├── model_training.py # ML model training
├── model_evaluation.py # Performance metrics
├── visualization.py # Data visualization
├── utils.py # Utility functions
├── docker-compose.yml # Docker configuration
├── Dockerfile # Container specification
├── requirements.txt # Dependencies
└── README.md # Documentation


## 🎬 Technical Pipeline
1. **Data Collection**: Targeted tweet acquisition via Twitter API
2. **Preprocessing**: NLP pipeline including tokenization, lemmatization, and noise removal
3. **Sentiment Analysis**: ML-based classification using Logistic Regression
4. **Visualization**: Statistical analysis and trend visualization

## 📊 Performance Metrics
- Logistic Regression with CountVectorizer
- 80/20 train-test split
- Multi-class classification metrics
- Continuous model evaluation and refinement

## 🐳 Docker Implementation
- Isolated runtime environment
- Automated dependency management
- Volume mounting for persistent data
- Environment-based configuration
- Streamlined deployment process

## 🚨 Current Limitations
- Keyword-based sentiment baseline
- English language support only
- Twitter API rate limitations
- Basic sentiment classification model

## 🔮 Development Roadmap
- Enhanced ML model architecture
- Multilingual support implementation
- Real-time streaming capabilities
- Web interface development
- Advanced emotion classification

## 🤝 Contributing
Contributions welcome! Check out our contribution guidelines for:
- Feature implementations
- Bug fixes
- Documentation improvements
- Performance optimizations

## 📜 License
MIT License - See LICENSE file for details
