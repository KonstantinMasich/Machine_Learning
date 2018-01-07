import pandas as pd
ex = pd.read_excel('titanic3.xls')
print(ex.head())

# 1. Getting victims list -- only name + pclass
with open('d.csv') as f:
    victims = f.read().splitlines()
l = []
for row in victims:
    l.append(row.split()[:-1])

vic_list = []    
for row in l:
    del row[1] # Remove title
    del row[3] # Remove port
    row[-1] = row[-1][:-2]
    del row[2] # Remove age
    vic_list.append(' '.join(row))

df = pd.read_csv('t.csv')[['PassengerId', 'Name']]
df['Last Name']  = df['Name'].apply(lambda name: name.split(',')[0])
df['Name'] = df['Name'].apply(lambda name: ''.join(name.split(',')[1:]))
df['Name'] = df['Name'].apply(lambda name: name.strip().split('(', 1)[0].strip())
df['Name'] = df['Name'].apply(lambda name: name.split('.', 1)[1].replace('"', '').strip())
df['Pclass'] = pd.read_csv('t.csv')['Pclass']
df['Survived'] = 1 # 1 by default

index = 0
for l_name, name, p_class in zip(df['Last Name'], df['Name'], df['Pclass']):
    s = ' '.join([l_name, name, str(p_class)])
    if s in vic_list:
        df.set_value(index, 'Survived', 0)
    index += 1
print(df.head())

#df.to_csv('out.csv', index=False)
df[['PassengerId', 'Survived']].to_csv('out.csv', index=False)
