import pandas as pd
import numpy as np
from django.shortcuts import render
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


def index(request):
    if request.method == 'POST':
        file = request.FILES['file']

        alpha = float(request.POST.get('value'))

        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            df = df.dropna()

            df['X'] = pd.to_numeric(df['X'], errors='coerce')
            df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
            df = df.dropna()
            df = df.round(2)
            table = df.to_html(
                classes='table table-striped table-hover table-bordered table-sm table-responsive text-center')

            # вычисление среднеквадратичного отклонения по Y и X
            std_y = np.std(df['Y'])
            std_x = np.std(df['X'])

            # вычисление линейного коэффициента корреляции
            corr_coef = df['X'].corr(df['Y'])

            # определение уровня корреляции между признаками x и y
            corr_level = 'слабая'
            if abs(corr_coef) > 0.7:
                corr_level = 'сильная'
            elif abs(corr_coef) > 0.3:
                corr_level = 'средняя'

            # вычисление средней ошибки коэффициента корреляции
            stderr_corr_coef = (1 - corr_coef ** 2) / (np.sqrt(len(df)))

            # проверка коэффициента корреляции на значимость
            t_score = abs(corr_coef) * np.sqrt(len(df) - 2) / np.sqrt(1 - corr_coef ** 2)
            p_value = 2 * (1 - stats.t.cdf(t_score, df=df.shape[0] - 2))

            significance = p_value < 0.05

            # 2) Нахождение табличного значения
            # Для коэффициента корреляции Спирмена воспользуемся критерием Стьюдента
            df_len = len(df)
            t_alpha = stats.t.ppf(1 - alpha / 2, df_len - 2)

            # 3) Коэффициент Спирмена
            spearman_corr = df['X'].corr(df['Y'], method='spearman')

            # 4) Уровень между признаками
            corr_level2 = 'слабая'
            if abs(spearman_corr) > 0.7:
                corr_level2 = 'сильная'
            elif abs(spearman_corr) > 0.3:
                corr_level2 = 'средняя'

            # 5) Эластичность
            # Эластичность - это отношение процентного изменения в Y к процентному изменению в X.
            # Для расчета нам нужно усредненное значение X и Y.
            mean_y = df['Y'].mean()
            mean_x = df['X'].mean()
            elasticity = spearman_corr * (mean_y / mean_x)

            # 6) Средняя ошибка аппроксимации
            y_pred = spearman_corr * df['X']
            approx_error = np.mean(np.abs((df['Y'] - y_pred) / df['Y']))
            # Определяем функцию для аппроксимации (в данном случае линейную)
            def linear_func(x, a, b):
                return a * x + b

            # Получаем коэффициенты аппроксимирующей функции
            popt, pcov = curve_fit(linear_func, df['X'], df['Y'])

            # Считаем значения, полученные из аппроксимирующей функции
            approx_values = linear_func(df['X'], *popt)

            # Считаем разницу между значениями y и значениями, полученными из аппроксимирующей функции
            diff = df['Y'] - approx_values

            # Считаем среднее значение разностей
            #approx_error = np.mean(diff)

            # 7) Общая дисперсия

            total_var = np.sum((df['Y'] - mean_y)**2)/df_len

            # 8) Факторная дисперсия
            # Построение уравнений регрессии
            x1 = df['X'].values
            y2 = df['Y'].values

            a0,a1 = np.polyfit(x1, y2, 1)
            # Считаем среднее значение признака y для каждого уникального значения признака x
            x_f= df.round(0)
            mean_y_by_x = x_f.groupby('X')['Y'].mean()

            # Считаем среднее значение всех средних значений признака y
            mean_yy = np.mean(mean_y_by_x)
            np.round(mean_yy)
            # Вычисляем факторную дисперсию
            factor_var = np.sum((mean_y_by_x - mean_yy) ** 2) / df_len

            # 9) Остаточная дисперсия
            res_var = total_var - factor_var

            # 10) Проверка общей дисперсии
            # Если общая дисперсия равна сумме факторной и остаточной дисперсии, проверка пройдена
            var_check = np.isclose(total_var, factor_var + res_var)

            # 11) Теоретический коэффициент детерминации
            # Коэффициент детерминации R^2 - это квадрат коэффициента корреляции Спирмена
            determination_coef = spearman_corr ** 2

            # 1) Теоретическое корреляционное отношение
            # Это квадратный корень из коэффициента детерминации
            determination_coef = corr_coef ** 2
            corr_ratio = np.sqrt(determination_coef)

            # 2) Зависимость между коррелированными отношениями
            rel_level = 'слабая'
            if abs(corr_ratio) > 0.7:
                rel_level = 'сильная'
            elif abs(corr_ratio) > 0.3:
                rel_level = 'средняя'

            # 3) F-критерий Фишера
            f_score = determination_coef / (1 - determination_coef) * (len(df) - 2)

            # 4) При заданном уровне значимости считает Fтабличное
            f_critical = stats.f.ppf(1 - alpha, 1, len(df) - 2)

            # Убедимся, что X не равен 0 для гиперболической регрессии
            df = df[df['X'] != 0]

            # Построение уравнений регрессии
            x = df['X'].values
            y = df['Y'].values

            # Линейная регрессия
            linear_coefs = np.polyfit(x, y, 1)
            y_linear = np.polyval(linear_coefs, x)

            # Параболическая регрессия
            x_quadratic = np.linspace(x.min(), x.max(), 500)
            quadratic_coefs = np.polyfit(x, y, 2)
            y_quadratic = np.polyval(quadratic_coefs, x_quadratic)

            # Гиперболическая регрессия
            x_hyperbolic = np.linspace(x.min(), x.max(), 500)
            hyperbolic_coefs = np.polyfit(1 / x, y, 1)
            y_hyperbolic = np.polyval(hyperbolic_coefs, 1 / x_hyperbolic)

            plt.rcParams['lines.markersize'] = 0.8
            # Построение графиков
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))

            ax[0].scatter(x, y, color='blue')
            ax[0].plot(x, y_linear, color='red')
            ax[0].set_title('Линейная регрессия')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')

            ax[1].scatter(x, y, color='blue')
            ax[1].plot(x_quadratic, y_quadratic, color='red')
            ax[1].set_title('Параболическая регрессия')
            ax[1].set_xlabel('X')
            ax[1].set_ylabel('Y')

            ax[2].scatter(x, y, color='blue')
            ax[2].plot(x_hyperbolic, y_hyperbolic, color='red')
            ax[2].set_title('Гиперболическая регрессия')
            ax[2].set_xlabel('X')
            ax[2].set_ylabel('Y')

            plt.tight_layout()
            plt.savefig('plot.svg')

            return render(request, 'success.html', {
                'table': table,
                'alpha': alpha,
                't_alpha': t_alpha,
                'spearman_corr': spearman_corr,
                'corr_level': corr_level,
                'corr_level2': corr_level2,
                'elasticity': elasticity,
                'approx_error': approx_error,
                'total_var': total_var,
                'factor_var': factor_var,
                'res_var': res_var,
                'var_check': var_check,
                'determination_coef': determination_coef,
                'corr_ratio': corr_ratio,
                'std_y': std_y,
                'std_x': std_x,
                'corr_coef': corr_coef,
                'stderr_corr_coef': stderr_corr_coef,
                't_score': t_score,
                'p_value': p_value,
                'significance': significance,
                'rel_level': rel_level,
                'f_score': f_score,
                'f_critical': f_critical,

            })

        else:
            return render(request, 'upload.html')
    else:
        return render(request, 'upload.html')