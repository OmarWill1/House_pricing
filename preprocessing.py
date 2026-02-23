import pandas as pd 


data_frame = pd.read_csv("Housing.csv")
data_frame = data_frame.sample(frac=1, random_state=42).reset_index(drop=True)

data_frame['price'] = data_frame['price'] / 1000 
print(data_frame.info())
print(data_frame.describe())

# removig outliers 

q1 = data_frame['price'].quantile(0.25)
q3 = data_frame['price'].quantile(0.75)

iqr = q3 - q1 

# the bounds 

upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr 

filtered_df = data_frame[ (data_frame['price'] >= lower_bound ) & (data_frame['price'] <= upper_bound) ]

print(f"Original rows {len(data_frame)}")
print(f"rows after filtering {len(filtered_df)}")
print(f"the iqr is {iqr}")



print(filtered_df.head().describe())

print(filtered_df.select_dtypes(include='object'))

for column in filtered_df.select_dtypes(include='object').columns :
    print(f"{column} : {filtered_df[column].unique()} , lenght of {len(filtered_df[column].unique())}")
    if len(filtered_df[column].unique()) ==2 :
        filtered_df[column] = filtered_df[column].map({'yes' : 1 , 'no' : 0})
        
    else :
        filtered_df = pd.get_dummies(filtered_df , columns=[column] , drop_first=True , dtype=int)


#feature engineering 
## creates a feature that ONLY exists for high-end homes.
filtered_df['total_rooms'] = (filtered_df['bedrooms'] + filtered_df['bathrooms'] ) * filtered_df['stories']
filtered_df['area_per_room']  = filtered_df['area'] / filtered_df['total_rooms']

filtered_df['area_squared'] = filtered_df['area'] ** 2 


filtered_df = filtered_df.drop(columns=['area_per_room' , 'total_rooms'])
corr_matrix = filtered_df.corr()
correlation_price = filtered_df.corr(numeric_only=True)["price"].sort_values(ascending=False)
filtered_corr = correlation_price[abs(correlation_price) > 0.25 ]
columns = filtered_corr.index
filtered_df = filtered_df[columns]
print(correlation_price)
print(columns)


#drop the outlier





filtered_df.to_csv("cleaned_data.csv" , index=False)
columns = filtered_df.drop(columns=['price']).columns

