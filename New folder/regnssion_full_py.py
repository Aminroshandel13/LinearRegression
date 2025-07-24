
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

tips=sns.load_dataset('tips')
tips_encoded = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)

X=tips_encoded.drop('tip', axis=1)
y=tips_encoded['tip']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

print("ضریب total_bill:", model.coef_[0])
print("ضریب size:", model.coef_[1])
print("مقدار ثابت (intercept):", model.intercept_)

coef_names = X.columns
for name, coef in zip(coef_names, model.coef_):
    print(f"ضریب {name}: {coef:.2f}")
print(f"مقدار ثابت (intercept): {model.intercept_:.2f}")

# 7. رسم نمودار مقایسه مقدار واقعی و پیش‌بینی‌شده
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:20], label='tip واقعی', marker='o')
plt.plot(y_pred[:20], label='tip پیش‌بینی‌شده', marker='x')
plt.title('مقایسه مقدار واقعی و پیش‌بینی‌شده tip (20 نمونه اول)')
plt.xlabel('نمونه')
plt.ylabel('tip')
plt.legend()
plt.grid(True)
plt.show()

