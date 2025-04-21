# Restaurant Review Sentiment Analysis with GUI

## Project Overview
This project implements a sophisticated restaurant review sentiment analyzer with a graphical user interface (GUI). It combines traditional machine learning with modern AI capabilities through DeepSeek's language model to provide comprehensive sentiment analysis of restaurant reviews.

## Key Features

### 1. Dual Analysis System
- **Traditional Machine Learning Model**
  - Uses Bag of Words (BoW) for text vectorization
  - Employs a trained classifier for sentiment prediction
  - Provides confidence scores for predictions

- **DeepSeek AI Integration**
  - Utilizes DeepSeek's advanced language model
  - Offers more nuanced sentiment understanding
  - Provides modern AI-powered analysis

### 2. Graphical User Interface
- Clean and intuitive interface built with Tkinter
- Features:
  - Large text input area for reviews
  - Model selection options (Traditional/DeepSeek/Both)
  - Sample review buttons for quick testing
  - Color-coded results (green for positive, red for negative)
  - Separate result panels for each analysis method

### 3. Text Processing Pipeline
1. Text Cleaning
   - Removes special characters
   - Converts to lowercase
   - Tokenizes the text

2. Advanced Processing
   - Stopword removal (preserves negation words)
   - Word stemming using Porter Stemmer
   - Vectorization for traditional model

## Technical Details

### Dependencies
- Python 3.6 or higher
- Required packages:
  - `tkinter` for GUI
  - `nltk` for text processing
  - `scikit-learn` for traditional model
  - `numpy` and `pandas` for data handling
  - `requests` for API communication
  - `python-dotenv` for environment management

### Models
1. **Traditional Model**
   - Uses pre-trained Bag of Words model (`c1_BoW_Sentiment_Model.pkl`)
   - Classifier model (`c2_Classifier_Sentiment_Model`)
   - Binary classification (0 for negative, 1 for positive)

2. **DeepSeek Integration**
   - Connects to DeepSeek's API
   - Uses environment variables for secure API key storage
   - Returns binary sentiment classification

## Usage Instructions

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure DeepSeek API:
   - Create `.env` file
   - Add your DeepSeek API key
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

### Running the Application
1. Launch the application:
   ```bash
   python sentiment_analyzer_gui.py
   ```
2. Enter a restaurant review in the text box
3. Select analysis method:
   - Traditional: Uses only the machine learning model
   - DeepSeek: Uses only the AI model
   - Both: Shows results from both models
4. Click "Analyze Sentiment" or use sample reviews

### Interface Elements
- **Text Input**: Large area for entering reviews
- **Model Selection**: Radio buttons for choosing analysis method
- **Sample Reviews**: Pre-loaded examples of positive and negative reviews
- **Results Display**: 
  - Separate panels for each model
  - Confidence scores (for traditional model)
  - Color-coded results for easy interpretation

## Error Handling
- Validates API key configuration
- Handles network errors for API calls
- Provides clear error messages in the GUI
- Graceful handling of missing model files

## Project Structure
```
├── sentiment_analyzer_gui.py    # Main application file
├── requirements.txt            # Package dependencies
├── .env                       # Environment variables
├── c1_BoW_Sentiment_Model.pkl # Traditional model files
└── c2_Classifier_Sentiment_Model
```

## Future Enhancements
- Batch processing of multiple reviews
- Sentiment trend analysis
- Export functionality for results
- More detailed sentiment analysis (beyond binary classification)
- Integration with restaurant review platforms

## Technical Notes
- The GUI is built using Tkinter for cross-platform compatibility
- Asynchronous API calls prevent GUI freezing
- Modular design allows easy model swapping
- Environment variables ensure secure API key management

## Conclusion
This project demonstrates the power of combining traditional machine learning with modern AI capabilities in a user-friendly interface. The dual analysis system provides robust sentiment analysis for restaurant reviews, making it a valuable tool for restaurant owners and customers alike. 