import scipy.stats as stats
import numpy as np

class TextAnalyzer:
    def analyze_features(self, texts, sentiments):
        # Word count analysis
        word_counts = [len(text.split()) for text in texts]
        sentiment_correlation = stats.pointbiserialr(word_counts, sentiments)
        
        # Feature importance
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Chi-square test for feature selection
        chi2_scores, p_values = chi2(X, sentiments)
        important_features = [(feature, score) for feature, score 
                            in zip(feature_names, chi2_scores)]
        
        return {
            'word_count_correlation': sentiment_correlation,
            'important_features': sorted(important_features, 
                                      key=lambda x: x[1], 
                                      reverse=True)[:10]
        } 

    def predict(self, X):
        if not hasattr(self, 'best_model'):
            raise ValueError("Model needs to be trained first")
        return self.best_model.predict(X)