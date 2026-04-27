import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from Preprocessing import Preprocessing

nltk.download('vader_lexicon', quiet=True)


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.sia = SentimentIntensityAnalyzer()

        self.urgency_words = [
            "urgent", "asap", "immediately", "deadline",
            "today", "tomorrow", "important", "priority",
            "submit", "meeting", "action required",
            "time sensitive", "by eod", "by end of day",
            "overdue", "past due", "please respond"
        ]


    def urgency_score(self, text):
        if not isinstance(text, str):
            return 0
        count = 0
        text = text.lower()
        for word in self.urgency_words:
            if word in text:
                count += 1
        return count


    def sentiment_score(self, text):
        if not isinstance(text, str):
            return 0
        scores = self.sia.polarity_scores(text)
        return scores['compound']


    def subject_score(self, subject):
        important_words = [
            "meeting", "project", "deadline", "submission",
            "interview", "exam", "urgent", "action required",
            "important", "asap", "review", "approval"
        ]
        if not isinstance(subject, str):
            return 0
        score = 0
        subject = subject.lower()
        for word in important_words:
            if word in subject:
                score += 1
        return score


    def sender_score(self, sender):
        if not isinstance(sender, str):
            return 0
        sender = sender.lower()
        # Spam / automated senders — lowest priority
        if any(x in sender for x in ["noreply", "no-reply", "newsletter",
                                      "unsubscribe", "donotreply", "mailer"]):
            return 0
        # High-importance senders
        if any(x in sender for x in ["manager", "director", "ceo", "cto",
                                      "professor", "faculty", ".edu", "admin"]):
            return 3
        # Default real person
        return 2


    def thread_score(self, subject):
        if not isinstance(subject, str):
            return 0
        subject = subject.lower()
        # Re: and Fwd: mean the original action was already taken
        # penalise slightly so fresh emails rank higher
        if "re:" in subject or "fwd:" in subject or "fw:" in subject:
            return -1
        return 0

    def time_score(self, date_string):
        if not isinstance(date_string, str):
            return 0
        try:
            email_time = pd.to_datetime(date_string, utc=True)
            hour = email_time.hour
            if 9 <= hour <= 18:   # work hours → more important
                return 2
            else:
                return 1
        except Exception:
            return 0

    def priority_label(self, row):
        score = (
            row["urgency_score"] * 3 +    # strongest signal
            row["subject_score"] * 2 +    # subject keywords matter a lot
            row["sender_score"]  * 2 +    # who sent it matters
            row["sentiment_score"] * 1 +  # -1 to 1, softer influence
            row["time_score"]    * 1 +    # work hours boost
            row["thread_score"]  * 1      # penalises replies/fwds slightly
        )
        if score >= 8:
            return "High"
        elif score >= 4:
            return "Medium"
        else:
            return "Low"

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
        # Final priority label — must come after all score columns are set
        self.df["priority"] = self.df.apply(self.priority_label, axis=1)
        return self.df

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
                    "time_score",
                    "priority"
                ]
            ].head(n)
        )

    def show_priority_distribution(self):
        counts = self.df["priority"].value_counts()
        total  = len(self.df)
        print("\n── Priority Distribution ──")
        for label in ["High", "Medium", "Low"]:
            count = counts.get(label, 0)
            bar   = "█" * int(count / total * 30)
            print(f"  {label:<8} {count:>5}  {bar}")
        print()


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
    fe = FeatureExtractor(p.df)
    final_df = fe.extract_features()

    fe.show_features()
    fe.show_priority_distribution()