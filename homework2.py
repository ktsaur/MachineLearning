import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

def main():
    # Загрузка
    df = pd.read_csv('AmesHousing.csv')
    df = df.drop(columns=['Order', 'PID'], errors='ignore')
    df = df.dropna(subset=['SalePrice'])

    # Обработка
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna("NA")
    df = pd.get_dummies(df, drop_first=True)

    # Удаляем коррелирующие признаки
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df_cleaned = df.drop(columns=to_drop)
    X = df_cleaned
    y = df['SalePrice']

    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['SalePrice'] = y

    # 3D-график: x, y — признаки (PCA), z — SalePrice
    fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='SalePrice',
                        color='SalePrice', color_continuous_scale='Viridis',
                        title='3D PCA: признаки -> SalePrice')
    fig.show()

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Подбор alpha и метрика RMSE
    alphas = np.logspace(-4, 1, 20)
    rmse_list = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    # График зависимости RMSE от alpha
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, rmse_list, marker='o')
    plt.xscale('log')
    plt.xlabel('Коэффициент регуляризации (alpha)')
    plt.ylabel('RMSE')
    plt.title('Зависимость RMSE от alpha (Lasso)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Лучшая модель
    best_alpha = alphas[np.argmin(rmse_list)]
    print(f'Лучшее значение alpha: {best_alpha:.4f}, минимальный RMSE: {min(rmse_list):.2f}')

    # Признаки и важность
    best_model = Lasso(alpha=best_alpha, max_iter=10000)
    best_model.fit(X_scaled, y)
    feature_importance = pd.Series(best_model.coef_, index=X.columns).sort_values(key=abs, ascending=False)

    print(feature_importance.head(5))

if __name__ == '__main__':
    main()