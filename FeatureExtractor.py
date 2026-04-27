import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
from Preprocessing import Preprocessing

nltk.download('vader_lexicon', quiet=True)


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()

        # urgency keywords
        self.urgency_words = [
            "urgent", "asap", "immediately", "deadline",
            "today", "tomorrow", "important", "priority",
            "submit", "meeting", "action required"
        ]

    # ---------------------------
    # Feature 1: Urgency Detection
    # ---------------------------
    def urgency_score(self, text):
        if not isinstance(text, str):
            return 0

        count = 0
        text = text.lower()

        for word in self.urgency_words:
            if word in text:
                count += 1

        return count

    # ---------------------------
    # Feature 2: Sentiment Analysis
    # ---------------------------
    def sentiment_score(self, text):
        if not isinstance(text, str):
            return 0

        scores = self.sia.polarity_scores(text)

        return scores['compound']

    # ---------------------------
    # Feature 3: Subject Importance
    # ---------------------------
    def subject_score(self, subject):
        important_words = [
            "meeting",
            "project",
            "deadline",
            "submission",
            "interview",
            "exam",
            "urgent"
        ]

        if not isinstance(subject, str):
            return 0

        score = 0
        subject = subject.lower()

        for word in important_words:
            if word in subject:
                score += 1

        return score

    # ---------------------------
    # Feature 4: Sender Importance
    # ---------------------------
    def sender_score(self, sender):
        if not isinstance(sender, str):
            return 0

        sender = sender.lower()

        if ".edu" in sender:
            return 3
        elif "manager" in sender:
            return 3
        elif "newsletter" in sender:
            return 1
        elif "noreply" in sender:
            return 0
        else:
            return 2

    # ---------------------------
    # Feature 5: Thread Depth
    # ---------------------------
    def thread_score(self, subject):
        if not isinstance(subject, str):
            return 0

        subject = subject.lower()

        score = 0

        if "re:" in subject:
            score += 1

        if "fwd:" in subject:
            score += 1

        return score

    # ---------------------------
    # Feature 6: Time Features
    # ---------------------------
    def time_score(self, date_string):
        if not isinstance(date_string, str):
            return 0

        try:
            email_time = pd.to_datetime(date_string)

            hour = email_time.hour

            # work hours = more important
            if 9 <= hour <= 18:
                return 2
            else:
                return 1

        except:
            return 0

    # ---------------------------
    # Apply all features
    # ---------------------------
    def extract_features(self):
        self.df["urgency_score"] = self.df["clean_body_classify"].apply(
            self.urgency_score
        )

        self.df["sentiment_score"] = self.df["clean_body_classify"].apply(
            self.sentiment_score
        )

        self.df["subject_score"] = self.df["subject"].apply(
            self.subject_score
        )

        self.df["sender_score"] = self.df["from"].apply(
            self.sender_score
        )

        self.df["thread_score"] = self.df["subject"].apply(
            self.thread_score
        )

        self.df["time_score"] = self.df["date"].apply(
            self.time_score
        )

        return self.df

    # ---------------------------
    # View sample output
    # ---------------------------
    def show_features(self, n=5):
        print(
            self.df[
                [
                    "subject",
                    "urgency_score",
                    "sentiment_score",
                    "subject_score",
                    "sender_score",
                    "thread_score",
                    "time_score"
                ]
            ].head(n)
        )


# --------------------------------
# Main Execution
# --------------------------------
if __name__ == "__main__":
    p = Preprocessing(sample_size=1000)

    print("Parsing emails...")
    p.apply_parse()

    print("Cleaning emails...")
    p.apply_cleaning()

    print("Tokenizing...")
    p.tokenization()

    print("Lemmatizing...")
    p.lemmatization()

    print("Extracting features...")
    feature_extractor = FeatureExtractor(p.df)

    final_df = feature_extractor.extract_features()

    feature_extractor.show_features()

