import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

class ProductPredictionPipeline:
    def __init__(self):
        pass

    def load_data(self, file_path):
        df = pd.read_csv(file_path, delimiter=';')
        df['Income'] = (df['Income'] * 1000).astype(int)
        df['Mortgage'] = (df['Mortgage'] * 1000).astype(int)
        return df

    def encode_categorical_columns(self, df, categorical_cols):
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])
        return df, label_encoder

    def assign_products(self, row, df):
        products = []
        yearly_income = row['Income'] * 12
        remaining_money = yearly_income - row['CCAvg_Yearly']
        money_left_is_half = remaining_money < (0.5 * yearly_income)

        if money_left_is_half:
            return 'Money Management Counseling'

        high_spending = row['CCAvg'] > 2500
        if 'Online Banking' not in products:
            products.append('Online Banking')

        if row['Income'] > 40000 and row['Income'] <= 80000 and row['Mortgage'] == 0:
            products.append('Personal Loan')
        if row['Income'] > 100000 and not high_spending:
            products.append('Securities Account')
        if 40000 < row['Income'] <= 70000 and row['Age'] > 50 and row['Mortgage'] < 50000:
            products.append('CD Account')
        if row['Income'] > 50000 and row['CCAvg'] < 1000:
            products.append('Credit Card')
        if high_spending and row['Income'] > 80000:
            products.append('Premium Credit Card')
        elif high_spending and row['Income'] <= 80000:
            products.append('Budget Adjustment Counseling')
        if row['Income'] > 60000 and row['Age'] >= 25 and row['Age'] <= 55 and row['Mortgage'] == 0:
            products.append('Mortgage Offer')

        offered_columns = ['Mortgage', 'Personal Loan', 'Securities Account', 'CD Account', 'Online Banking', 'Credit Card']
        for offered_column in offered_columns:
            if offered_column in df.columns and row[offered_column] > 0:
                products = [product for product in products if product != offered_column]

        return ', '.join(products) if products else 'Other'

    def preprocess_data(self, df):
        df['CCAvg_Yearly'] = df['CCAvg'] * 12
        df['Product'] = df.apply(lambda row: self.assign_products(row, df), axis=1)
        df = df[df['Product'] != 'Other']
        return df

    def train_model(self, df, feature_cols, target_col):
        y = df[target_col]
        X = df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def evaluate_model(self, model, X_test, y_test, label_encoder, df):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        classification_rep = classification_report(y_test, y_pred)

        client_ids = df.loc[X_test.index, 'ID']
        predicted_products = label_encoder.inverse_transform(y_pred)
        income = df.loc[X_test.index, 'Income']
        experience = df.loc[X_test.index, 'Experience']
        mortgage = df.loc[X_test.index, 'Mortgage']
        ccavg_yearly = df.loc[X_test.index, 'CCAvg_Yearly']
        ages = df.loc[X_test.index, 'Age']

        predictions_df = pd.DataFrame({
            'Client ID': client_ids,
            'Predicted Product(s)': predicted_products,
            'Income': income,
            'Age': ages,
            'Experience': experience,
            'Mortgage': mortgage,
            'CCAvg_Yearly': ccavg_yearly
        })

        predictions_df.to_csv('predictions_with_products_checked.csv', index=False)
        with open('model_evaluation_metrics.txt', 'w') as file:
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"F1-Score: {f1}\n")
            file.write("\nClassification Report:\n")
            file.write(classification_rep)

        return accuracy, f1, classification_rep

    def visualize_tree(self, model, feature_cols, label_encoder):
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_cols, class_names=label_encoder.classes_, filled=True)
        plt.show()

    def run_pipeline(self, file_path):
        df = self.load_data(file_path)
        categorical_cols = ['ExperienceCategory', 'IncomeCategory', 'FamilySizeCategory', 'SpendingCategory',
                            'MortgageDependency', 'AgeGroup', 'EducationLevel']
        df, label_encoder = self.encode_categorical_columns(df, categorical_cols)
        df = self.preprocess_data(df)
        df['Product_Label'] = label_encoder.fit_transform(df['Product'])

        feature_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage',
                        'ExperienceCategory', 'IncomeCategory', 'FamilySizeCategory',
                        'SpendingCategory', 'MortgageDependency', 'AgeGroup']
        model, X_test, y_test = self.train_model(df, feature_cols, 'Product_Label')
        accuracy, f1, classification_rep = self.evaluate_model(model, X_test, y_test, label_encoder, df)
        self.visualize_tree(model, feature_cols, label_encoder)

        print(f"Accuracy: {accuracy}")
        print(f"F1-Score: {f1}")
        print("Classification Report:")
        print(classification_rep)

# Run the pipeline
pipeline = ProductPredictionPipeline()
pipeline.run_pipeline('dataset/RBK_with_engineered_features.csv')
