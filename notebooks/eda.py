import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "../data/processed/processed_sales_data.csv"
data = pd.read_csv(data_path)

data.head()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Sales', data=data)
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()
