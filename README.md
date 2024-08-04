# Final project "Analyzing phone market"

## Part 1: Getting, parsing and saving the data

### 1. Імпортуємо бібліотеку requests (для звернення до веб-ресурсів), beautifulSoup (для конверсії відповіді response в об'єкт html-soup), sqlite3 - (для роботи з sqlite), pandas - (для роботи з датафреймами).
```python
import requests                   # бібліотека для здійснення запитів
from bs4 import BeautifulSoup     # для парсингу html
import sqlite3                    # для роботи з БД, зокрема SQlite
import pandas as pd               # для роботи з датафреймами
```

### 2. У якості використовуваного веб-ресурсу приймемо [сторінку з гаджетами](https://can.ua/smartphones/c1538/). Звернемося до неї та отримаємо першу сторінку:
```python
url_base = "https://can.ua/smartphones/c1538/" # початкова сторінка з данними
url_page="page=" # додаток до адреси для отримання сторінки

data = requests.get(url_base + url_page + str(1)) # отримання першої сторінки
```

### 3. Якщо сайт дозволив отримати дані (статус код = 200), то конвертуємо дані в об'єкт BeautifulSoup для подальшої роботи і створюємо базу даних для запису отриманих даних. В іншому випадку - сповіщаємо, який статус код прийшов користувачу і завершуємо роботу програми:
```python
if data.status_code == 200:
    print("Data was successfully retrieved (200)")
    
    soup = BeautifulSoup(data.text, "html.parser") # парсимо текст відповіді в об'єкт ГарногоСупа
    
    db = sqlite3.connect("cellphones.db") # створили базу даних з назвою cellphones, або підключились до існуючої (якщо вона була)
    cur = db.cursor() # отримали курсор (мідлуеар) для подальшого оперування запитами
    
    # створюємо таблицю cellphones, у якій будемо зберігати:
        # id_cellphone - унікальний артикул товару з сайту
        # title - повна назва товару
        # photo_ref - посилання на головну фотографію товару
        # price - ціна товару
        # model - модель товару
    cur.execute('''CREATE TABLE IF NOT EXISTS cellphones(
        id_cellphone INT NOT NULL PRIMARY KEY, 
        title VARCHAR(100) NOT NULL,
        photo_ref VARCHAR(255) NULL, 
        price INT NOT NULL, 
        model VARCHAR(100) NOT NULL ) ''')
         
    # створюємо таблицю characteristics, у якій будемо зберігати:
        # model_id - унікальний артикул товару з сайту, прив'язаний до таблиці cellphones
        # display - діагональ дісплею
        # display_size - розмір дісплею
        # cores_num - кількість ядер
        # ram_gb - оперативна пам'ять
        # IMC_gb - внутрішня пам'ять
        # camera_mp - мегапікселі камери
        # os - операційна система
    cur.execute('''CREATE TABLE IF NOT EXISTS characteristics(
        model_id INT NOT NULL PRIMARY KEY, 
        display FLOAT NOT NULL,
        display_size VARCHAR(45) NOT NULL,
        cores_num INT NOT NULL,
        ram_gb INT NOT NULL, 
        IMC_gb INT NOT NULL,
        camera_mp INT NOT NULL,
        os VARCHAR(45) NOT NULL, 
        FOREIGN KEY(model_id) REFERENCES cellphones(id_cellphone))''')
        
    db.commit() # commit the transaction (saves changes in DB)
    cur.close() # позначаємо програмі кінець поточної операції з курсором, зачиняючи його
    
else:
    print("Unable to retrieve the data. Response code = ", data.status_code)
### <center>Схема спроектованої бази даних з Power BI</center>
![db_scheme.png](attachment:db_scheme.png)
```

### 4. Якщо статус код = 200, то отримуємо кількість сторінок як максимальний елемент списку сторінок на сайті. Розпочинаємо цикл по всім сторінкам для отримання даних. Створюємо БД та таблиці, записуємо в них отримані дані. Для подробиць, будь ласка, зверніться до коментарів у програмі:
```python
if data.status_code == 200: # якщо сайт не проти
    print("Data was successfully retrieved (200)")
    
    # отримуємо кількість сторінок розділу
    pages_qty = int(soup.select_one("ul[name='paginator'] li:nth-last-child(2)").get_text()) 
    # отримуємо label'и зі списку моделей сайту
    model_labels = soup.select("#desktop_sort-producer > ul > li > label")
    models = [] # список моделей з сайту (для подальшого їх аналізу у назві товару)

    for model in model_labels: # для кожної моделі зі списку
        models.append(model.get_text().split("(")[0].strip()) # парсимо лише назву моделі у масив з моделями
    
    for i in range(1, pages_qty+1): # цикл по всім сторінкам
        print("Fulfilling page #", i, "... of", pages_qty) # виводимо номер поточної сторінки, щоб розуміти прогрес виконання програми
        if i != 1: # для першої сторінки нічого не робимо, тому що у якості тестового запиту вона вже була отримана
            data = requests.get(url_base + url_page + str(i)) # а для інших сторінок посилаємо запит на новую сторінку
        
        soup = BeautifulSoup(data.text, "html.parser") # парсимо текст відповіді в об'єкт ГарногоСупа
        
        page_products = soup.select(".catalog-tile li") # отримуємо всі товари з поточної (і) сторінки
        
        for product in page_products: # цикл по всіх продуктах поточної сторінки (за замовчуванням сайту 40 шт. на сторінці)
            id = product.select_one("div.info__code").get_text().strip()[5:] # унікальний артикул товару з сайту
            title = product.select_one("a.info__name").get_text()  # повна назва товару
            price = 0 # за замовчуванням - ціна 0
            price_element = product.select_one("p.item__price") # елемент, з якого читаємо ціну
            
            if price_element: # якщо такий елемент знайдено
                price = (price_element.get_text()[:-4]).replace(" ", '')  # ціна товару зберігається (відповідно, товар є в наявності)
            
            photo_ref = product.select_one("img")['src']  # посилання на фотографію товару
            model = "" # модель товару, що обмежена значеннями
            
            for curr_model in models: # цикл по всім моделям
                if curr_model in title: # перевіряємо кожну і зупиняємося, коли знайшли
                    model = curr_model # перед зупинкою зберігаємо значення, що знайшли
                    break # сама зупинка
            
            # секція "додаткові параметри"
            display = product.select_one(".item__hidden > p >b:nth-child(1)").get_text()[:-2]  # діагональ дісплею
            display_size = product.select_one(".item__hidden > p >b:nth-child(3)").get_text() # розмір дісплею
            cores_num = product.select_one(".item__hidden > p >b:nth-child(5)").get_text() # кількість ядер
            ram = product.select_one(".item__hidden > p >b:nth-child(7)").get_text()[:-3] # оперативна пам'ять
            imc = product.select_one(".item__hidden > p >b:nth-child(9)").get_text()[:-3] # внутрішня пам'ять
            camera = product.select_one(".item__hidden > p >b:nth-child(11)").get_text()[:-3] # мегапікселі камери
            os = product.select_one(".item__hidden > p >b:nth-child(13)").get_text() # операційна система
            
            cur = db.cursor() # оголошуємо курсор знову, позначаючи початок роботи із запитами
            # записали дані у першу таблицю
            try: # спробувати записати дані у БД
                cur.execute('INSERT INTO cellphones VALUES(?, ?, ?, ?, ?)', (id,title, photo_ref, price,model))
                # записали дані у другу таблицю
                cur.execute('INSERT INTO characteristics VALUES(?, ?, ?, ?, ?, ?, ?, ?)', (id, display,display_size,cores_num, ram, imc, camera, os))
            except sqlite3.IntegrityError as e: # якщо була помилка порушення цілісності
                if 'UNIQUE constraint' in e.args[0]: # якщо дублюється унікальний запис
                    print("Немає можливості додати запис(и), тому що вони порушують унікальність коду БД. Спробуйте спочатку видалити ці записи або просто видаліть БД та запустіть програму")
                else: # інша помилка
                    print("Error: ", e)

                break # при помилці не продовжувати парсинг сторінок і запис у БД
            else: # якщо все пройшло без помилок
                db.commit() # зберігаємо зміни
            finally: # виконувати у будь-якому разі
                cur.close() # позначаємо кінець поточних транзакцій
                
    db.close() # завершуємо роботу з базою даних        
    print("Done") # сповіщаємо користувача, що програма завершила роботу
else:
    print("Unable to retrieve data. Response code = ", data.status_code) # у разі помилки із самого початку
```

### 5. Отримуємо дані із БД:
```python
db = sqlite3.connect("cellphones.db") # підключаємося до БД cellphones.db
cur = db.cursor() # оголошуємо новий курсор

# за допомогою об'єднання таблиць отримуємо всі дані з БД
cur.execute('''SELECT id_cellphone, title, photo_ref, price, model, display, display_size, cores_num, ram_gb, IMC_gb, camera_mp, os 
                FROM cellphones 
                LEFT JOIN characteristics ON cellphones.id_cellphone = characteristics.model_id''')
df = pd.DataFrame(cur.fetchall(), columns=["Артикул", "Назва", "Фото", "Ціна", "Модель", "Дисплей", "Розмір дисплею", "Кількість ядер", "Оперативна пам'ять", "Внутрішня пам'ять", "Камера", "Операційка"])

cur.close() # кінець поточних транзакцій
db.close() # завершуємо роботу з зазою даних

df.head()
## Part 2: Visualization
import matplotlib.pyplot as plt    # import the library for visualizing
import seaborn as sns              # for plots styling
display(df.describe()) # describe the data
df.isnull().values.any() # check for nulls (if any)
# function to get the string value percentage on the column
def get_percentage_str(data_frame_column, value, round_to):
    return str(round((value/sum(data_frame_column))*100,round_to))+"%"

sns.set_style('darkgrid') # dark grid style

df_vis = df[df['Ціна'] > 0] # not showing rows with 0 prices (out of stock)

avg_price_per_model = df_vis[['Модель','Ціна']].groupby('Модель', as_index=False)['Ціна'].mean() # average price per model
avg_price_per_model = avg_price_per_model.sort_values('Ціна', ascending=False) # sorting by price from biggest to lowest
count_per_model = df_vis['Модель'].value_counts() # models count per model
count_per_model = pd.DataFrame({'Модель': count_per_model.index, 'Середня кількість': count_per_model.values})
avg_price_count_per_model = pd.merge(avg_price_per_model, count_per_model, on=['Модель', 'Модель'])
avg_price_count_per_model = avg_price_count_per_model.sort_values('Середня кількість', ascending=False)


fig1 = plt.figure(figsize=(12,14)) # the first figure to plot on
ax1 = fig1.add_subplot(2,2,1) # add subplot to the fig1 (2 rows, 2 columns, 1st plotting)
ax1.invert_yaxis() # invert y axis
ax1.xaxis.set_major_formatter(lambda x,p: str(int(x/1000))+"K") # custom format x axis labels

plt.barh(avg_price_per_model['Модель'].values, avg_price_per_model['Ціна'].values) # horizontal bar chart model by price
plt.xlabel('average price, grn', fontsize=16) # x axis label
plt.title("Models average cost ratio", fontsize = 18) # chart title



ax2 = fig1.add_subplot(2,2,2) # add subplot to the fig1 (2 rows, 2 columns, 2nd plotting)
ax2.invert_yaxis() # invert y axis
plt.barh(count_per_model['Модель'].values, count_per_model['Середня кількість'].values) # horizontal bar chart model by count
plt.xlabel('models, count', fontsize=16) # x axis label
plt.title("Models count ratio", fontsize = 18) # chart title

fig1.tight_layout() # for the plots not to overlap



fig2 = plt.figure(figsize=(10,6)) # the second figure to plot on
fig2.suptitle("Models' averaged prices by count", fontsize = 20) # figure 2 title
ax3 = fig2.add_subplot(1,1,1) # add subplot to the fig2 (1 row, 1 column, 1st plotting)
ax3.yaxis.set_major_formatter(lambda x,p: str(int(x/1000))+"K") # custom format y axis labels
# scatter plot count by price
plt.scatter(avg_price_count_per_model['Середня кількість'].values, # x-values: average count per model
            avg_price_count_per_model['Ціна'].values, # y-values: average price per model
            c=[i for i in range(1, len(count_per_model.values)+1)]) # separate color for each model
plt.xlabel('models, count', fontsize=16) # x axis label
plt.ylabel('average price, grn', fontsize=16) # y axis label

# filter models to get the most popular and/or expensive models
models_filtered = avg_price_count_per_model[((avg_price_count_per_model['Ціна']>14000.0) & # prices bigger than 14K AND
                                             (avg_price_count_per_model['Середня кількість']<20.0)) # count less that 20 OR
                                            | (avg_price_count_per_model['Середня кількість']>37)] # count bigger than 37

for i, txt in enumerate(list(models_filtered['Модель'].values)): # for every filtered model
    # render labels in the custom formatting: model_name+count_percentage
    ax3.annotate(txt + "\n(" + get_percentage_str(models_filtered['Середня кількість'], models_filtered['Середня кількість'].values[i], 2) +")", 
                 (models_filtered['Середня кількість'].values[i]+1, # label x-val (+1 for better visual)
                  models_filtered['Ціна'].values[i]-800),  # label y-val (-800 for better visual)
                 size=11) # font-size

fig2.tight_layout() # for the plots not to overlap



fig3 = plt.figure(figsize=(12,10)) # the third figure to plot on
fig3.suptitle('TOP 5 models', fontsize=20) # figure 3 title
ax4 = fig3.add_subplot(3,2,1) # add subplot to the fig3 (3 rows, 2 columns, 1st plotting)
_, __, texts = plt.pie(avg_price_per_model['Ціна'].head().values, # the first 5 most expensive average price values per model
                       colors=['#43A4E6', '#12239E', '#E66C37', '#6B007B', '#E044A7'], # the colors for each model
                       autopct='%1.2f%%', # auto-estimation of percent in the float formatting
                       #custom value format for labels: model_name+(price/1000)K
                       labels=[k + " " + str(round(int(v)/1000,2))+"K" for k,v in avg_price_per_model.head().values],
                       textprops={'fontsize':10}) # size of the labels
    
for text in texts: # for each of labels of the plt.pie
    text.set_horizontalalignment('center') # align labels center
    text.set_color('white') # font color
    text.set_weight('bold') # font weight
    
plt.title("Top 5 models (average price)", fontsize = 18) # the pie plot title



ax5 = fig3.add_subplot(3,2,2)  # add subplot to the fig3 (3 rows, 2 columns, 2nd plotting)
_, texts = plt.pie(count_per_model['Середня кількість'].head().values, # the first 5 biggest average count values per model
                   colors=['#43A4E6', '#12239E', '#E66C37', '#6B007B', '#E044A7'],  # the colors for each model
                   startangle=100, # starting angle for the plot to be shown
                   labeldistance=0.6, # distance between labels and the pie center
                   # custom value format for labels: model_name+<br>+(percent_from_the_whole)%
                   labels=[str(k)+"\n("+ get_percentage_str(count_per_model['Середня кількість'], v, 2) +")" 
                           for k,v in count_per_model.head().values], 
                   textprops={'fontsize':10}) # size of the labels

for text in texts: # for each of labels of the plt.pie
    text.set_horizontalalignment('center')  # align labels center
    text.set_color('white') # font color
    text.set_weight('bold') # font weight

plt.title("Top 5 models (count)", fontsize = 18) # the pie plot title
# the pie plot legend
plt.legend(labels=count_per_model['Модель'].head().values, # labels are the five top models
           loc='upper center', # legend location
           bbox_to_anchor=(0.5, -0.04), # the corner's location
           ncol=3) # number of columns

fig3.tight_layout() # for the plots not to overlap



plt.show() # show the plots
```

## Part 3: Analyzing and model learning
### Стоїть задача вибору телефону. Фінансів вистачає на найдорожчу модель у магазині, але доцільність і релевантність витрати стоїть на першому місці. 
### Тож, постає необхідність вирішення задачі максимізації користі телефону за його ціну. Необхідними підзадачами до виконання є:
<ul><font size="4">
<li>врахування репутації та популярності на ринку кожної моделі (якщо їх занадто багато, взяти n найкращих);</li>
<li>зрозуміти принципи, за якими можна поділити всі телефони та розглядати їх, як окремі групи (що більше за все впливає на ціну телефону);</li>
<li>за допомогою найдоцільнішого методу навчити мережу розпізнавати модель телефону за відповідними характеристиками (включаючи ціну); перевірити здатність системи на реальних та майже- або не- реальних прикладах.</li>
</font></ul>

```python
import random                                                                         # working with pseudo-random values
from sklearn.cluster import KMeans                                                    # kmeans exclusive clustering alg-thm
from sklearn.linear_model import LogisticRegression                                   # logistic regression alg-thm
from sklearn import tree                                                              # decision tree alg-thm
from sklearn.ensemble import RandomForestClassifier                                   # random forest alg-thm
from xgboost import XGBClassifier                                                     # XGB alg-thm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score   # analyzing learning results
from sklearn.model_selection import train_test_split                                  # splitting data for train/test
from sklearn.ensemble import VotingClassifier                                         # ensemble to use multiple alg-thms
display(df.head())
display(df.describe())
df = df.drop(['Артикул', 'Назва', 'Фото'], axis=1) # getting rid of useless columns
 # splitting display size by "(" to standrdize it, removing whitespaces
df['Розмір дисплею'] = df['Розмір дисплею'].apply(lambda x: x.split('(')[0].strip())

display(df['Розмір дисплею'].unique()) # unique display sizes
df.head()
df = pd.get_dummies(df, columns=['Розмір дисплею']) # divide 'Розмір дисплею' column into separate columns

display(df.head())
display(df['Операційка'].unique()) # unique values of OSes
df['Операційка'] = df['Операційка'].apply(lambda x: 0 if x == 'Android (без поддержки Google Services та Play Market)' or x == 'Android' else (1 if x == 'iOS' else 'unfamiliar model'))

display(df['Операційка'].unique())
display(df.head())
display(df.describe())
df_learn = df.copy() # dataframe for alg-thms to be learnt

display(df_learn.shape) # sizes (rows, columns)

df_learn = df_learn.drop(df_learn[df_learn['Ціна']==0].index) # drop 0 prices (out of stock; no info)

display(df_learn.shape) # sizes (rows, columns)
display(df_learn.head())
model_counts = df_learn['Модель'].value_counts() # counts per model

model_counts
df_learn = df_learn[df_learn['Модель'].isin(model_counts.head().index)] # leave only the top 5 most popular models
unique_models = df_learn['Модель'].unique() # check all the unique models in df for learning

unique_models
price_0_models = df[df['Ціна']==0] # save 0 prices to the separate df
# leave there the same models as in df for learning
price_0_models = price_0_models[price_0_models['Модель'].isin(unique_models)]

price_0_models.head()
# std of price per model
std_price_per_model = df_learn[['Модель','Ціна']].groupby('Модель', as_index=False)['Ціна'].std()
# in avg price leave the same models as in learning df
avg_price_per_model = avg_price_per_model[avg_price_per_model['Модель'].isin(unique_models)]

display(avg_price_per_model)
display(std_price_per_model)
for model in unique_models: # for every top model
    curr_model_std = std_price_per_model[std_price_per_model['Модель']==model]['Ціна'].values[0] # curr model's price std
    curr_model_avg = avg_price_per_model[avg_price_per_model['Модель'] == model]['Ціна'].values[0] # curr model's price avg
    # for every price in 0 price df change price into average price for the model +- its std price
    price_0_models.loc[price_0_models['Модель']==model, 'Ціна'] = price_0_models[price_0_models['Модель']==model]['Ціна']\
    .apply(lambda x: curr_model_avg + random.uniform(-curr_model_std, curr_model_std))
    
display(price_0_models.head())
display(df_learn.head())
df_learn['Модель'] = df_learn['Модель'].apply(lambda x: list(unique_models).index(x)) # encode models as indexes
df_to_predict_res = price_0_models['Модель'].apply(lambda x: list(unique_models).index(x)) # encode models for 0 models also
df_to_predict = price_0_models.drop('Модель',axis=1) # delete value for prediction from the test df

display(df_learn.head())
display(df_to_predict.head())
display(df_to_predict_res.head())
try:
    # trying to parse all columns in all dfs as floats to check for any remaining string-ish elements
    df_learn = df_learn.astype('float')
    df_to_predict = df_to_predict.astype('float')
    df_to_predict_res = df_to_predict_res.astype('float')
except Exception as e: # in case of error
    print(e) # print the error
finally: # anyway
    print("Done")
distortions = [] # list for appending inertias for the elbow method
for k in range(1,11): # estimate kmeans for different number of clusters (from 1 to 10)
    kmean = KMeans(n_clusters=k, # amount of clusters
               n_init=15 # number of times to repeat algorithm to find better centroids
              ).fit(df_learn)

    distortions.append(kmean.inertia_) # Sum of squared distances of samples to their closest cluster center
    
# plotting
plt.plot([i for i in range(1,11)], # x values
         distortions) # y values
plt.xlabel('k, clusters', fontsize=18) # x axis label
plt.ylabel('Weighted errors', fontsize=16) # y axis label

plt.show()
# we're choosing between
kmeans_2_cl = KMeans(n_clusters=2, n_init=15).fit(df_learn) # 2 clusters kmeans
kmeans_3_cl = KMeans(n_clusters=3, n_init=15).fit(df_learn) # 3 clusters kmeans

df_2_clusters, df_3_clusters = df_learn.copy(), df_learn.copy() # copy df for learning for each kmeans
df_2_clusters['Cluster'] = kmeans_2_cl.labels_ # create the 'Cluster' column to associate each row (model) with its cluster
df_3_clusters['Cluster'] = kmeans_3_cl.labels_

df_grouped_2 = df_2_clusters.groupby('Cluster', as_index=False).mean() # group by cluster and get means (for analysis)
df_grouped_3 = df_3_clusters.groupby('Cluster', as_index=False).mean()

display(df_grouped_2)
display(df_grouped_2.describe())
display(df_grouped_3)
display(df_grouped_3.describe())
# plotting clusters with c different colors for each separate cluster
sns.reset_orig() # get back to standard bg


fig = plt.figure(figsize=(14,10)) # the first figure to plot on
plt.subplot(2, 2, 1) # add subplot (2 rows, 2 columns, 1st plotting)
# scatter plot price by device memory colored separately for each cluster, half transparent (2 clusters)
plt.scatter(df_2_clusters['Ціна'], df_2_clusters["Внутрішня пам'ять"], c=df_2_clusters['Cluster'], alpha=.5)
plt.title("2 Clusters", fontsize=20) # the scatter plot title
plt.xlabel('Ціна', fontsize=16) # x axis label
plt.ylabel("Внутрішня пам'ять", fontsize=16) # y axis label



plt.subplot(2, 2, 2) # add subplot (2 rows, 2 columns, 2nd plotting)
# scatter plot price by device memory colored separately for each cluster, half transparent (3 clusters)
plt.scatter(df_3_clusters['Ціна'], df_3_clusters["Внутрішня пам'ять"], c=df_3_clusters['Cluster'], alpha=.5)
plt.title("3 Clusters", fontsize=20) # the scatter plot title
plt.xlabel('Ціна', fontsize=16) # x axis label
plt.ylabel("Внутрішня пам'ять", fontsize=16) # y axis label



plt.subplot(2, 2, 3) # add subplot (2 rows, 2 columns, 3rd plotting)
# scatter plot price by camera colored separately for each cluster, half transparent (2 clusters)
plt.scatter(df_2_clusters['Ціна'], df_2_clusters["Камера"], c=df_2_clusters['Cluster'], alpha=.5)
plt.title("2 Clusters", fontsize=20) # the scatter plot title
plt.xlabel('Ціна', fontsize=16) # x axis label
plt.ylabel("Камера", fontsize=16) # y axis label



plt.subplot(2, 2, 4) # add subplot (2 rows, 2 columns, 4th plotting)
# scatter plot price by camera colored separately for each cluster, half transparent (3 clusters)
plt.scatter(df_3_clusters['Ціна'], df_3_clusters["Камера"], c=df_3_clusters['Cluster'], alpha=.5)
plt.title("3 Clusters", fontsize=20) # the scatter plot title
plt.xlabel('Ціна', fontsize=16) # x axis label
plt.ylabel("Камера", fontsize=16) # y axis label

fig.tight_layout() # for the plots not to overlap


plt.show() # show the plots
fig = plt.figure(figsize=(16,10)) # the first figure to plot on
ax = plt.subplot(1,1,1) # add subplot (1 row, 1 column, 1st plotting)
# scatter plot price by device memory colored separately for each cluster, half transparent (3 clusters)
ax.scatter(df_3_clusters['Ціна'], df_3_clusters["Внутрішня пам'ять"], c=df_3_clusters['Модель'], alpha=.5)
ax.xaxis.set_major_formatter(lambda x,p: str(int(x/1000))+"K") # custom format y axis labels
plt.xlabel('Ціна', fontsize=16) # x axis label
plt.ylabel("Внутрішня пам'ять", fontsize=16) # y axis label

plt.show() # show the plot
# dividing df for learning into separate dfs
X = df_learn.drop('Модель', axis=1).astype('float') # independent (x) values

Y = df_learn['Модель'] # dependent value (y) for predicting

X_train, X_test, y_train, y_test = train_test_split(X, # independent variables
                                                    Y, # dependent variable (expected output)
                                                    random_state=1, # pseudo-random number parameter to reproduce the same train test split each time you run the code 
                                                    train_size = .75) # 75% is aimed for training, other 25% - for testing

# дерево рішень
# easy model with least data preparation. Based on multiple statistical choice,
# but even slight difference in data can significantly impact the final solution
# - calculate gini impurity for all impure leafes (that are of ambigious choice)
# for every independent variable: 1 - (probability of true)^2 - (probability of false)^2
# (for non-binary values we calculate average of every pair and check if the current value
# less or bigger than the current average)
# - for the root element of the tree the least gini impurity is chosen
# - for further leafs reduce impurity by adding other calculated leafs
# - recalculate gini impurity for elements that were filtered by the root elements
decision_tree = tree.DecisionTreeClassifier().fit(X_train, y_train) # fit the splitted data from df_learn
dtree_predictions = decision_tree.predict(X_test) # test how good is this algorythm

print("Accuracy of Dicision Tree: {0}%".format(accuracy_score(y_test, dtree_predictions) * 100))
print(confusion_matrix(y_test, dtree_predictions)) 
print(classification_report(y_test, dtree_predictions))
# логістична регресія
# logistic regression has a sygmoid to fit the data rather than a stright line
# it also predicts binary values, not discrete ones, lika the linear regression
# instead of minimizing sum of residual squares (we can also get rid off non-helping
# independent variables):
# 1. pick a probability, scaled by weight
# 2. using it calculate the likelihood of the prediction being "False"
# 3. do this for all data and multiply them - that is the likelihood of the data
# according to the current sigmoid
# 4. shift the line and repeat 1-4 until we reach the maximum likelihood
fit_lg = LogisticRegression(penalty='none').fit(X_train, y_train) # fit the splitted data from df_learn
lg_predictions = fit_lg.predict(X_test) # test how good is this algorythm

print("Accuracy of Logistic Regression: {0}%".format(accuracy_score(y_test, lg_predictions) * 100))
print(confusion_matrix(y_test, lg_predictions))
print(classification_report(y_test, lg_predictions))
# random forest
# similar to the Tree Clasifier, but it may build hundreds of trees randomly
# and then, predictin the data, we run our test data through every tree and
# then venture the decision according to the most votes
rf = RandomForestClassifier(n_estimators = 800, random_state = 41).fit(X_train, y_train) # fit the splitted data from df_learn
rf_predictions = rf.predict(X_test) # test how good is this algorythm
accuracy_score_rf = accuracy_score(y_test, rf_predictions)

print("Accuracy of Logistic Regression: {0}%".format(accuracy_score_rf * 100))
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
#XGBoost trees
# basically ensemble of trees, built on top of each other, each correcting the latter. Algorythm includes:
# gradient boost, regularization, a un ique regression tree, approximate greedy algorithm,
# wighted quantile sketch, sparsity-aware split finding, parallel learning,
# cache-aware access, blocks out of core computation
model_XGB = XGBClassifier().fit(X_train, y_train)
XG_predictions = model_XGB.predict(X_test)
accuracy_score_XG = accuracy_score(y_test, XG_predictions)
print("Accuracy of Logistic Regression: {0}%".format(accuracy_score_XG * 100))
print(confusion_matrix(y_test, XG_predictions))
print(classification_report(y_test, XG_predictions))
# trying out ensemble
estimators = [("RForest", rf), ("XGB", model_XGB)] # set of algorythms to combine
ensemble = VotingClassifier(estimators, # the algorythms
                            voting='hard') # make predicitions by majority vote

ensemble.fit(X_train, y_train) # fit the splitted data from df_learn
Y_prediction = ensemble.predict(X_test) # test how good is this algorythm

print(classification_report(Y_prediction,y_test))
print("Voting Ensemble:>" + str(accuracy_score(Y_prediction,y_test)*100) + "%")
predictions = rf.predict(df_to_predict) # predicting 0 prices models with pseudo-randomized prices using the best alg

print(classification_report(predictions,df_to_predict_res))
print("Result:>" + str(accuracy_score(predictions,df_to_predict_res)*100) + "%")
avg_price_per_model
# minimizing prices, maximizing attributes (for each OS)
predictions = rf.predict([[5250.974359, 7.6, 8, 12, 512, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [5250.974359, 7.6, 8, 12, 512, 200, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                         ]) 

display(predictions)
display(unique_models)
```

## Summary
### Врахувавши популярність кожної моделі, вибір було звужено до наступних п'яти:
```python
avg_price_per_model['Модель']
```

#### Ціна будь-якої моделі сильно залежить від камери, внутрішньої пам'яті та бренду. Всі моделі  було вирішено поділити на сегмент бюджетний:
```python
df_grouped_3[df_grouped_3['Cluster']==2]
середній:
df_grouped_3[df_grouped_3['Cluster']==1]
дорогий:
df_grouped_3[df_grouped_3['Cluster']==0]
```

<font size="3">
    &emsp;Найдоцільнішим методом навчання мережі виявився <b>Random Forest</b>. Для данної моделі не підійшли б методи чисельного наближення або пошуку загальних закономірностей, адже бюджетний сегмент має дуже високу щільність значень, частина з яких не пов'язана по групах.<br>
&emsp;Точність навченої моделі склала <b>97.59036144578313%</b> для справжньої вибірки і <b>79.47368421052632%</b> для імітованої.<br>
&emsp;Тож, довіряючи системі можемо вирішити задачу мінімізації ціни при максимізації параметрів відповідно для кожної з ОС:
<ul>
    <li>Для Android найкращим вибором буде модель SAMSUNG;</li>
    <li>Для iOS найкращим вибором буде модель Apple;</li>
</ul>
&emsp;Оскільки, на основі досліджених даних для Apple переплата йде значною кількістю за бренд (найвище середнє значення ціни при досить великій кількості моделей), ця альтернатива виключається.<br>
&emsp;<b>Фінальний вибір залишається за моделлю SAMSUNG.</b>
</font>
