# TweetFeel: Social Media Sentiment Analysis

## Problem Definition
Social media sentiment analysis presents challenges in understanding public opinion due to:
- Informal language and slang
- Sarcasm and context-dependent meanings
- Real-time processing requirements

## AI Implementation
### Branch: Natural Language Processing & Machine Learning
- **Algorithm**: Logistic Regression
- **Feature Extraction**: CountVectorizer
- **Text Processing**: NLTK for tokenization and lemmatization

### Design Decisions
1. **Model Choice**: 
   - Logistic Regression chosen for interpretability
   - Suitable for text classification tasks
   - Good performance with limited data

2. **Feature Engineering**:
   - Text cleaning (URLs, mentions removal)
   - Stop word removal
   - Lemmatization for word normalization

3. **Evaluation Metrics**:
   - Precision, Recall, F1-score
   - Multi-class classification report 