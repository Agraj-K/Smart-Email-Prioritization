import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from Preprocessing import Preprocessing
from FeatureExtractor import FeatureExtractor


MODEL_FILE = "priority_model.pkl"


class PriorityClassifier:
    def __init__(self):
        print("[Stage 5] Running Stage 1 → Stage 3 pipeline...")

        p = Preprocessing(sample_size=1000)

        print("Parsing emails...")
        p.apply_parse()

        print("Cleaning emails...")
        p.apply_cleaning()

        print("Tokenizing...")
        p.tokenization()

        print("Lemmatizing...")
        p.lemmatization()

        print("\nRunning BERT Feature Extraction...")
        fe = FeatureExtractor(p.df)

        self.df = fe.extract_features()

        print("\nStage 3 completed successfully.")

    # -----------------------------------
    # Better Label Generation
    # -----------------------------------
    def create_labels(self):
        def assign_priority(row):
            score = 0

            score += row["urgency_score"] * 2
            score += row["action_score"] * 2
            score += row["sender_score"]
            score += row["thread_score"]

            if row["sentiment_score"] < -0.5:
                score += 1

            if score >= 8:
                return "High"
            elif score >= 4:
                return "Medium"
            else:
                return "Low"

        self.df["priority_label"] = self.df.apply(
            assign_priority,
            axis=1
        )

        print("\nPriority Distribution:")
        print(self.df["priority_label"].value_counts())

    # -----------------------------------
    # Train Model
    # -----------------------------------
    def train_model(self):
        print("\nPreparing training data...")

        bert_cols = [
            col for col in self.df.columns
            if col.startswith("bert_")
        ]

        structured_cols = [
            "urgency_score",
            "action_score",
            "sentiment_score",
            "sender_score",
            "thread_score"
        ]

        feature_cols = bert_cols + structured_cols

        X = self.df[feature_cols]
        y = self.df["priority_label"]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42
        )

        print("\nTraining XGBoost classifier...")

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax"
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\nAccuracy:")
        print(accuracy_score(y_test, predictions))

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        joblib.dump(model, MODEL_FILE)
        print(f"\nModel saved as {MODEL_FILE}")

        self.model = model
        self.encoder = encoder
        self.feature_cols = feature_cols

    # -----------------------------------
    # Sample Predictions
    # -----------------------------------
    def sample_predictions(self, n=5):
        sample = self.df.head(n).copy()

        predictions = self.model.predict(
            sample[self.feature_cols]
        )

        sample["Predicted Priority"] = (
            self.encoder.inverse_transform(predictions)
        )

        print("\nSample Predictions:")
        print(
            sample[
                [
                    "subject",
                    "clean_body_summary",
                    "Predicted Priority"
                ]
            ]
        )


if __name__ == "__main__":
    pc = PriorityClassifier()

    pc.create_labels()
    pc.train_model()
    pc.sample_predictions()