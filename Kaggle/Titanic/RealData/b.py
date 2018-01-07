import pandas as pd
test_df = pd.read_csv('t.csv')[['PassengerId', 'Name']]  # Data in test file
real_df = pd.read_excel('titanic3.xls')[['survived', 'name']] # "Real" data
#real_df['PassengerId'] = -1
#print(real_df.head())

# Delete quote marks:
real_df['name'].replace('"','',regex=True,inplace=True)
test_df['Name'].replace('"','',regex=True,inplace=True)
#print(test_df.at[384, 'Name'] == real_df.at[595, 'name'])

survived = []
for name, p_id in zip(test_df['Name'], test_df['PassengerId']):
    row = real_df.loc[real_df['name'] == name]
    survived.append(row['survived'].iloc[0])
print(len(survived))
test_df['Survived'] = survived
#print(test_df.head())

test_df[['PassengerId', 'Survived']].to_csv('out.csv', index=False)
