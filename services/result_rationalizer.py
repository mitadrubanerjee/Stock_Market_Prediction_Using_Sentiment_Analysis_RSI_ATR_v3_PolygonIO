def rationalize_result(overall_sentiment, prediction):
    """
    Rationalizes the final output based on the overall sentiment and prediction.
    
    Args:
        overall_sentiment (str): The overall sentiment category, such as "Negative," "Trending Negative," "Neutral," 
                                 "Positive," or "Trending Positive."
        prediction (str): The model's prediction, either "upward" or "downward".

    Returns:
        str: A message rationalizing the sentiment and prediction.
    """

    # Set sentiment categories based on the overall sentiment string
    if overall_sentiment in ["Negative", "Trending Negative"]:
        sentiment_category = "Negative & Trending Negative"
    elif overall_sentiment in ["Positive", "Trending Positive"]:
        sentiment_category = "Positive & Trending Positive"
    else:
        sentiment_category = "Neutral"
    
    # Rationalize the prediction based on the sentiment category
    if prediction == "downward":
        if sentiment_category == "Negative & Trending Negative":
            return "As the Sentiment is mostly Negative, the model is predicting a general downward trend."
        elif sentiment_category == "Neutral":
            return "Sentiment is mostly Neutral, even then the model is predicting a more cautious downward trend."
        elif sentiment_category == "Positive & Trending Positive":
            return "Even though the Sentiment is largely Positive, the Model is predicting a cautious downward trend."
    
    elif prediction == "upward":
        if sentiment_category == "Negative & Trending Negative":
            return "As the Sentiment is mostly Negative, the model is predicting a cautious upward trend."
        elif sentiment_category == "Neutral":
            return "Sentiment is mostly Neutral, even then the model is predicting a more cautious upward trend."
        elif sentiment_category == "Positive & Trending Positive":
            return "As the Sentiment is largely Positive, the Model is predicting a general upward trend."

    # Default message if something goes wrong (should not reach here if inputs are correct)
    return "A Proper Prediction could not be made. Please check the inputs."
